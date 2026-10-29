#!/bin/bash
#
# Incremental-coverage runner. Default mode: build with coverage
# instrumentation, run the configured test filter, then produce a coverage
# report restricted to the source files (src/ + include/) changed relative
# to a baseline ref. Tests-only mode (--no-coverage): build without
# instrumentation, run the suites, print the test results - no baseline
# resolution, no lcov, no reports.
#
# The headline metric is strict changed-lines coverage: a line counts only
# if it was added/modified in `git diff -U0 <merge-base>` (working tree
# included, untracked files count as fully new) AND its lcov DA counter is
# non-zero. Overall per-file coverage of the changed files is printed as a
# reference row.
#
# Usage:
#   bash scripts/coverage/run_ut.sh [options]
#
# Options:
#   --base REF        baseline branch/commit (default: first of
#                     origin/main, origin/master, main, master that resolves)
#   --filter EXPR     Catch2 filter applied to every suite
#                     (default: "[ut][simd],[ft][hgraph],[ft][rabitq_split],"
#                              "[ft][ivf][rabitq]")
#   --timeout SECS    wall-clock limit per suite (default: 3600; 0 disables;
#                     skipped with a warning if the `timeout` command is absent)
#   --threshold N     changed-lines coverage gate in percent (default: 80)
#   --cmake-args STR  extra flags forwarded to the cmake configure step
#   --skip-build      reuse existing build/ (coverage mode: must already be
#                     configured with -DENABLE_COVERAGE=ON)
#   --no-html         skip the genhtml report
#   --no-coverage     tests-only mode: build without coverage instrumentation,
#                     run the suites and print the results; --base/--threshold/
#                     --no-html are ignored
#   -h, --help        show this help
#
# Examples:
#   bash scripts/coverage/run_ut.sh
#   bash scripts/coverage/run_ut.sh --base origin/1.0 --threshold 95
#   bash scripts/coverage/run_ut.sh --filter '[hgraph][rabitq]'
#   bash scripts/coverage/run_ut.sh --no-coverage
#
# Outputs (coverage mode, under coverage/):
#   incremental.info           lcov data restricted to changed files
#   incremental_report.txt     per-file changed-lines table + missed lines
#   incremental_missed.txt     every un-hit changed line (file:line)
#   incremental-html/          genhtml report of the changed files
#
# Exit code: 0 if all suites passed AND (tests-only mode OR changed-lines
# coverage >= threshold), else 1.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
COVERAGE_DIR="${ROOT_DIR}/coverage"
BASE=""
THRESHOLD=80
FILTER='[ut][simd],[ft][hgraph],[ft][rabitq_split],[ft][ivf][rabitq]'
TEST_TIMEOUT=3600
SKIP_BUILD=0
GEN_HTML=1
COVERAGE=1
CMAKE_EXTRA=()

usage() {
    # Print the leading comment block only (skip the shebang, stop at code).
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)       BASE="$2"; shift 2 ;;
        --filter)     FILTER="$2"; shift 2 ;;
        --timeout)    TEST_TIMEOUT="$2"; shift 2 ;;
        --threshold)  THRESHOLD="$2"; shift 2 ;;
        --cmake-args) read -r -a CMAKE_EXTRA <<< "$2"; shift 2 ;;
        --skip-build)  SKIP_BUILD=1; shift ;;
        --no-html)     GEN_HTML=0; shift ;;
        --no-coverage) COVERAGE=0; shift ;;
        -h|--help)    usage 0 ;;
        *)            echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

cd "${ROOT_DIR}"

[[ "${TEST_TIMEOUT}" =~ ^[0-9]+$ ]] ||
    { echo "ERROR: --timeout must be a non-negative integer (got '${TEST_TIMEOUT}')" >&2; exit 1; }

# Wall-clock guard: a single stuck test case must not hang the whole run
# forever. `timeout` is optional so this still works without coreutils.
HAVE_TIMEOUT=0
if [[ "${TEST_TIMEOUT}" -gt 0 ]]; then
    if command -v timeout >/dev/null 2>&1; then
        HAVE_TIMEOUT=1
    else
        echo "WARNING: 'timeout' not found - suites run without a wall-clock limit" >&2
    fi
fi

# Step labels: coverage mode runs steps 0..6; tests-only mode collapses to 3.
if [[ ${COVERAGE} -eq 1 ]]; then
    TOTAL_STEPS=6; STEP_RUN=3; STEP_SUMMARY=6; COVERAGE_FLAG=ON
else
    TOTAL_STEPS=3; STEP_RUN=2; STEP_SUMMARY=3; COVERAGE_FLAG=OFF
fi
step() { echo "=== [$1/${TOTAL_STEPS}] $2 ==="; }

for tool in cmake git; do
    command -v "${tool}" >/dev/null 2>&1 || { echo "ERROR: ${tool} not found" >&2; exit 1; }
done
if [[ ${COVERAGE} -eq 1 ]]; then
    command -v lcov >/dev/null 2>&1 || { echo "ERROR: lcov not found" >&2; exit 1; }
    if [[ ${GEN_HTML} -eq 1 ]]; then
        command -v genhtml >/dev/null 2>&1 || { echo "ERROR: genhtml not found" >&2; exit 1; }
    fi
fi

# ---------------------------------------------------------------------------
# 0. Resolve baseline and the changed source files (src/ + include/ only).
#    Diff is merge-base -> working tree, so uncommitted work is included.
#    Coverage mode only - tests-only runs need no baseline.
# ---------------------------------------------------------------------------
if [[ ${COVERAGE} -eq 1 ]]; then
    if [[ -z "${BASE}" ]]; then
        for ref in origin/main origin/master main master; do
            if git rev-parse --verify -q "${ref}^{commit}" >/dev/null 2>&1; then
                BASE="${ref}"
                break
            fi
        done
    fi
    [[ -n "${BASE}" ]] || { echo "ERROR: no baseline ref found; pass --base REF" >&2; exit 1; }
    MERGE_BASE="$(git merge-base "${BASE}" HEAD)"

    declare -a CHANGED=()      # repo-relative paths (relative to ROOT_DIR)
    declare -A KIND=()         # path -> "all" (fully new) | "hunk" (modified)
    mapfile -t DIFF_STATUS < <(git diff --name-status --diff-filter=ACMR "${MERGE_BASE}" -- src include)
    for entry in "${DIFF_STATUS[@]}"; do
        f="${entry##*$'\t'}"   # last field: new path for renames/copies
        case "${f}" in
            *.cpp|*.h|*.hpp) ;;
            *) continue ;;
        esac
        status="${entry%%$'\t'*}"   # A | M | R100 | C75 ...
        CHANGED+=("${f}")
        if [[ "${status}" == A* || "${status}" == R* || "${status}" == C* ]]; then
            KIND["${f}"]="all"
        else
            KIND["${f}"]="hunk"
        fi
    done
    mapfile -t UNTRACKED < <(git ls-files --others --exclude-standard -- src include)
    for f in "${UNTRACKED[@]}"; do
        case "${f}" in
            *.cpp|*.h|*.hpp) CHANGED+=("${f}"); KIND["${f}"]="all" ;;
        esac
    done

    step 0 "Baseline: ${BASE} (merge-base $(git rev-parse --short "${MERGE_BASE}")), changed source files: ${#CHANGED[@]}"
    if [[ ${#CHANGED[@]} -eq 0 ]]; then
        echo "No changed files under src/ or include/ - nothing to measure."
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# 1. Build (coverage mode mirrors `make cov`: Debug + ENABLE_COVERAGE;
#    tests-only mode drops the instrumentation for faster builds and runs).
# ---------------------------------------------------------------------------
if [[ ${SKIP_BUILD} -eq 0 ]]; then
    JOBS="$(nproc 2>/dev/null || echo 6)"
    if [[ ${COVERAGE} -eq 1 ]]; then
        step 1 "Configuring and building with coverage instrumentation"
    else
        step 1 "Configuring and building (tests-only, no coverage instrumentation)"
    fi
    # Log the environment so CI failures can be compared against local runs
    # (arch/compiler mismatches are the classic "works locally, breaks in CI").
    echo "build env: $(uname -srm)"
    # -w mirrors `make test`: ENABLE_WERROR defaults to ON upstream, so any
    # warning in code paths that only the CI machine compiles (e.g. the SVE
    # paths in simd/sve.cpp on aarch64) would become a CI-only build error.
    cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DENABLE_COVERAGE=${COVERAGE_FLAG} \
        -DENABLE_ASAN=OFF \
        -DENABLE_CCACHE=ON \
        -DENABLE_TESTS=ON \
        -DENABLE_MOCKIMPL=ON \
        -DCMAKE_CXX_FLAGS="-w" \
        "${CMAKE_EXTRA[@]}"
    cmake --build "${BUILD_DIR}" --parallel "${JOBS}"
else
    step 1 "Skipping build (--skip-build)"
fi
[[ -d "${BUILD_DIR}" ]] || { echo "ERROR: build dir ${BUILD_DIR} does not exist" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. Reset counters (.gcda files accumulate across runs). Coverage mode only.
# ---------------------------------------------------------------------------
if [[ ${COVERAGE} -eq 1 ]]; then
    step 2 "Clearing stale .gcda files"
    GCDA_COUNT="$(find "${BUILD_DIR}" -name '*.gcda' | wc -l)"
    [[ "${GCDA_COUNT}" -gt 0 ]] && find "${BUILD_DIR}" -name '*.gcda' -delete
    echo "removed ${GCDA_COUNT} stale .gcda file(s)"
fi

# ---------------------------------------------------------------------------
# 3. Run the suites with the filter.
# ---------------------------------------------------------------------------
step "${STEP_RUN}" "Running test suites (filter: ${FILTER})"
ANY_SUITE_FAILED=0
declare -a SUITE_REPORT=()

run_suite() {
    local name="$1" bin="$2" status rc=0
    if [[ ! -x "${bin}" ]]; then
        SUITE_REPORT+=("${name}: BINARY MISSING (${bin})")
        ANY_SUITE_FAILED=1
        return
    fi
    local -a cmd=("${bin}" -d yes "${FILTER}" --allow-running-no-tests)
    local limit="no limit"
    if [[ ${HAVE_TIMEOUT} -eq 1 ]]; then
        cmd=(timeout --kill-after=60s "${TEST_TIMEOUT}" "${cmd[@]}")
        limit="${TEST_TIMEOUT}s"
    fi
    echo "--- ${name}: ${bin} (timeout: ${limit}) ---"
    "${cmd[@]}" || rc=$?
    case ${rc} in
        0)      status=PASS ;;
        124|137) status="TIMEOUT (>${TEST_TIMEOUT}s)"
                 echo "ERROR: ${name} exceeded ${TEST_TIMEOUT}s and was killed" >&2 ;;
        *)      status="FAIL (exit ${rc})" ;;
    esac
    SUITE_REPORT+=("${name}: ${status}")
    [[ "${status}" == PASS ]] || ANY_SUITE_FAILED=1
}

run_suite "unittests"      "${BUILD_DIR}/tests/unittests"
run_suite "functests"      "${BUILD_DIR}/tests/functests"
run_suite "tests_mockimpl" "${BUILD_DIR}/mockimpl/tests_mockimpl"

# ---------------------------------------------------------------------------
# Tests-only mode ends here: report the results and exit.
# ---------------------------------------------------------------------------
if [[ ${COVERAGE} -eq 0 ]]; then
    step "${STEP_SUMMARY}" "Test results"
    echo ""
    echo "==================== TEST RESULTS ===================="
    for line in "${SUITE_REPORT[@]}"; do
        echo "  suite  : ${line}"
    done
    echo "  filter : ${FILTER}"
    echo "======================================================"
    if [[ ${ANY_SUITE_FAILED} -eq 1 ]]; then
        echo "RESULT: FAILED - at least one test suite failed"
        exit 1
    fi
    echo "RESULT: PASSED - all test suites passed"
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. Collect full coverage (project script; writes coverage/coverage.info)
#    and restrict it to the changed files -> coverage/incremental.info.
# ---------------------------------------------------------------------------
step 4 "Collecting coverage and extracting changed files"
if ! bash "${ROOT_DIR}/scripts/coverage/collect_cpp_coverage.sh" \
        > "${BUILD_DIR}/collect_coverage.log" 2>&1; then
    echo "ERROR: collect_cpp_coverage.sh failed - tail of ${BUILD_DIR}/collect_coverage.log:" >&2
    tail -30 "${BUILD_DIR}/collect_coverage.log" >&2
    exit 1
fi
INFO="${COVERAGE_DIR}/coverage.info"
[[ -s "${INFO}" ]] || { echo "ERROR: ${INFO} missing or empty" >&2; exit 1; }

INCR_INFO="${COVERAGE_DIR}/incremental.info"
lcov --extract "${INFO}" "${CHANGED[@]}" \
     --ignore-errors inconsistent,inconsistent,unused \
     --output-file "${INCR_INFO}"
if [[ ! -s "${INCR_INFO}" ]]; then
    echo "WARNING: none of the changed files has coverage records (never executed or not instrumented)"
    : > "${INCR_INFO}"
fi

# ---------------------------------------------------------------------------
# 5. Strict changed-lines metric: intersect lcov DA records with the
#    added/modified line ranges from `git diff -U0`.
# ---------------------------------------------------------------------------
step 5 "Computing changed-lines coverage"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
RANGES="${WORK}/ranges.tsv"

for f in "${CHANGED[@]}"; do
    if [[ "${KIND["${f}"]:-hunk}" == "all" ]]; then
        printf '%s\t1 99999999\n' "${f}" >> "${RANGES}"
    else
        # Zero-width sentinel: guarantees the file gets a report row even if
        # every hunk is a pure deletion (new-side count = 0). lo > hi never
        # matches a line number, it only registers the file in awk.
        printf '%s\t1 0\n' "${f}" >> "${RANGES}"
        git diff -U0 "${MERGE_BASE}" -- "${f}" | awk -v f="${f}" '
            /^@@ / {
                l = $0
                if (sub(/^@@ -[0-9]+(,[0-9]+)? \+/, "", l)) {
                    split(l, p, /[, ]/)
                    start = p[1] + 0
                    count = (p[2] ~ /^[0-9]+$/) ? p[2] + 0 : 1
                    if (count > 0)
                        printf "%s\t%d %d\n", f, start, start + count - 1
                }
            }' >> "${RANGES}"
    fi
done

RESULTS="${COVERAGE_DIR}/incremental_report.txt"
MISSED="${COVERAGE_DIR}/incremental_missed.txt"

# One row per changed file: file, changed-executable-lines, hit, pct,
# missed-lines (capped), seen-in-coverage (1/0). TOTAL row has no misses.
awk -v RANGES="${RANGES}" '
    BEGIN {
        while ((getline line < RANGES) > 0) {
            split(line, a, "\t"); f = a[1]; split(a[2], r, " ")
            n[f]++; lo[f, n[f]] = r[1]; hi[f, n[f]] = r[2]
        }
    }
    /^SF:/ { cur = substr($0, 4); if (cur in n) seen[cur] = 1 }
    /^DA:/ {
        if (!(cur in n)) next
        split(substr($0, 4), d, ",")
        ln = d[1] + 0
        for (i = 1; i <= n[cur]; i++) {
            if (ln >= lo[cur, i] && ln <= hi[cur, i]) {
                tot[cur]++
                if (d[2] + 0 > 0) hit[cur]++
                else {
                    miss[cur]++
                    if (miss[cur] <= 200) ml[cur] = ml[cur] (miss[cur] > 1 ? "," : "") ln
                }
                break
            }
        }
    }
    END {
        # missed-lines field must never be empty: IFS=$'\t' read collapses
        # consecutive tab whitespace, so an empty field shifts the columns.
        for (f in n) {
            t = tot[f] + 0; h = hit[f] + 0
            pct = (t > 0) ? sprintf("%.1f", 100 * h / t) : "n/a"
            printf "%s\t%d\t%d\t%s\t%s\t%d\n", f, t, h, pct, (ml[f] == "" ? "-" : ml[f]), seen[f] + 0
        }
        t = 0; h = 0
        for (f in tot) { t += tot[f]; h += hit[f] }
        if (t > 0) printf "TOTAL\t%d\t%d\t%.1f\t-\t1\n", t, h, 100 * h / t
    }' "${INCR_INFO}" > "${WORK}/results.tsv"

# Split rows by the seen flag (portable; no grep -P), sort seen rows worst-first.
awk -F'\t' '$NF == 1 && $1 != "TOTAL"' "${WORK}/results.tsv" \
    | sort -t$'\t' -k4,4n > "${WORK}/seen.tsv" || true
awk -F'\t' '$NF == 0' "${WORK}/results.tsv" > "${WORK}/absent.tsv"

: > "${MISSED}"
{
    echo "incremental coverage report ($(date '+%F %T'))"
    echo "baseline: ${BASE} (merge-base $(git rev-parse --short "${MERGE_BASE}"))"
    echo "filter:  ${FILTER}"
    echo
    printf '%-58s %8s %8s %8s\n' "changed file" "lines" "hit" "cover"
    while IFS=$'\t' read -r f t h pct ml _; do
        printf '%-58s %8d %8d %7s%%\n' "${f}" "${t}" "${h}" "${pct}"
        if [[ "${ml}" != "-" && -n "${ml}" ]]; then
            tr ',' '\n' <<< "${ml}" | sed "s|^|${f}:|" >> "${MISSED}"
        fi
    done < "${WORK}/seen.tsv"
    if [[ -s "${WORK}/absent.tsv" ]]; then
        echo
        echo "files with NO coverage records (never executed or not instrumented):"
        cut -f1 "${WORK}/absent.tsv" | sed 's/^/  /'
    fi
    if grep -q '^TOTAL' "${WORK}/results.tsv"; then
        echo
        echo "TOTAL changed executable lines: $(grep '^TOTAL' "${WORK}/results.tsv" | cut -f2), hit: $(grep '^TOTAL' "${WORK}/results.tsv" | cut -f3)"
    fi
} > "${RESULTS}"

STRICT_RATE="0.0"
TOTAL_ROW="$(grep '^TOTAL' "${WORK}/results.tsv" || true)"
[[ -n "${TOTAL_ROW}" ]] && STRICT_RATE="$(cut -f4 <<< "${TOTAL_ROW}")"
ABSENT_COUNT="$(wc -l < "${WORK}/absent.tsv")"

if [[ ${GEN_HTML} -eq 1 && -s "${INCR_INFO}" ]]; then
    genhtml --rc branch_coverage=1 \
        --ignore-errors inconsistent,inconsistent \
        -o "${COVERAGE_DIR}/incremental-html" "${INCR_INFO}" >/dev/null
fi

# ---------------------------------------------------------------------------
# 6. Gate on strict changed-lines coverage.
# ---------------------------------------------------------------------------
step 6 "Incremental coverage gate"
echo ""
echo "==================== INCREMENTAL COVERAGE ===================="
for line in "${SUITE_REPORT[@]}"; do
    echo "  suite           : ${line}"
done
echo "  baseline        : ${BASE} ($(git rev-parse --short "${MERGE_BASE}"))"
echo "  changed files   : ${#CHANGED[@]} under src/ + include/"
echo "  changed lines   : ${STRICT_RATE}%  (gate: >= ${THRESHOLD}%)"
[[ "${ABSENT_COUNT}" -gt 0 ]] && echo "  WARNING         : ${ABSENT_COUNT} changed file(s) have no coverage records"
echo "  report          : ${RESULTS}"
echo "  missed lines    : ${MISSED}"
[[ ${GEN_HTML} -eq 1 ]] && echo "  html            : ${COVERAGE_DIR}/incremental-html/index.html"
echo "=============================================================="

RC=0
if awk -v r="${STRICT_RATE}" -v t="${THRESHOLD}" 'BEGIN { exit !(r + 0 >= t + 0) }'; then
    echo "RESULT: incremental coverage gate PASSED (${STRICT_RATE}% >= ${THRESHOLD}%)"
else
    echo "RESULT: incremental coverage gate FAILED (${STRICT_RATE}% < ${THRESHOLD}%)"
    RC=1
fi
[[ ${ANY_SUITE_FAILED} -eq 1 ]] && { echo "RESULT: at least one test suite failed"; RC=1; }
exit ${RC}
