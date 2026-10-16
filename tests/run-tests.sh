#!/usr/bin/env bash

# run-tests.sh — Run Buddhabrot AVX-512 Benchmarks & Tests
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Find SBCL with AVX-512 support
if [ -z "${SBCL:-}" ]; then
  if [ -x "${ROOT_DIR}/../../sb-simd-512/run-sbcl.sh" ]; then
    SBCL="${ROOT_DIR}/../../sb-simd-512/run-sbcl.sh"
  elif [ -x "${ROOT_DIR}/../sb-simd-512/run-sbcl.sh" ]; then
    SBCL="${ROOT_DIR}/../sb-simd-512/run-sbcl.sh"
  elif [ -n "${HOME:-}" ] && [ -x "${HOME}/repos/sb-simd-512/run-sbcl.sh" ]; then
    SBCL="${HOME}/repos/sb-simd-512/run-sbcl.sh"
  elif command -v sbcl >/dev/null 2>&1; then
    SBCL="sbcl"
  else
    echo "Error: SBCL executable not found." >&2
    exit 1
  fi
fi

DYNAMIC_SPACE="${DYNAMIC_SPACE:-8000}"

usage() {
  echo "Usage: $0 [threads|registers|all]"
  echo ""
  echo "Commands:"
  echo "  threads     Run thread scaling benchmark (4, 8, 10, 12, 16, 20 threads)"
  echo "  registers   Run register saturation & ILP benchmark (1x vs 2x vs 3x vectors)"
  echo "  all         Run both benchmarks (default)"
  exit 1
}

TARGET="${1:-all}"

# Ensure buddha-512.fasl exists
if [ ! -f "${ROOT_DIR}/buddha-512.fasl" ]; then
  echo "==> Compiling buddha-512.lisp..."
  (cd "${ROOT_DIR}" && make compile)
fi

run_threads() {
  echo ""
  echo "========================================================================"
  echo " Running Thread Scaling Benchmark"
  echo "========================================================================"
  "${SBCL}" --dynamic-space-size "${DYNAMIC_SPACE}" --noinform \
    --script "${SCRIPT_DIR}/bench-threads.lisp"
}

run_registers() {
  echo ""
  echo "========================================================================"
  echo " Running Register Saturation & ILP Benchmark"
  echo "========================================================================"
  "${SBCL}" --dynamic-space-size "${DYNAMIC_SPACE}" --noinform \
    --script "${SCRIPT_DIR}/bench-registers.lisp"
}

case "${TARGET}" in
  threads|thread)
    run_threads
    ;;
  registers|register|regs)
    run_registers
    ;;
  all)
    run_threads
    run_registers
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown target: ${TARGET}"
    usage
    ;;
esac
