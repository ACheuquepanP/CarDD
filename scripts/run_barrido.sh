#!/usr/bin/env bash
# Barrido de ablación · secuencial (una fila a la vez) · tolerante a fallos.
# En la terminal muestra solo la curva de validación (mask_AP/box_AP por epoch);
# el log completo de cada fila queda en outputs/<nombre>_train.log
#
# Uso:            bash scripts/run_barrido.sh
# En background:  nohup bash scripts/run_barrido.sh > outputs/barrido.log 2>&1 &

set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src

FILAS=(
  "base_dcnNECK_cascade_ms   abl_base"
  "pointrend                 abl_pointrend"
  "cldice                    abl_cldice"
  "skeleton                  abl_skeleton"
  "pointrend_cldice          abl_pointrend_cldice"
  "pointrend_skeleton        abl_pointrend_skeleton"
  "cldice_skeleton           abl_cldice_skeleton"
  "test_stack_full           abl_full"
)

fallidas=()

for fila in "${FILAS[@]}"; do
  set -- $fila
  cfg="configs/ablation/custom/$1.yaml"
  log="outputs/$2_train.log"
  echo ""
  echo "=================================================================="
  echo ">>> [$1]  START $(date '+%H:%M:%S')"
  echo "=================================================================="

  # entrena en background, log completo al archivo
  python scripts/train.py --config "$cfg" > "$log" 2>&1 &
  pid=$!

  # mientras entrena, va mostrando en la terminal solo las líneas de validación
  # (mask_AP / box_AP por epoch) y los avisos de best.pth
  tail -n +1 -f --pid="$pid" "$log" 2>/dev/null | grep --line-buffered -E "val ·|nuevo best|early stopping" &
  tailpid=$!

  wait "$pid"
  rc=$?
  kill "$tailpid" 2>/dev/null

  if [ "$rc" -ne 0 ]; then
    echo ">>> [$1]  FALLÓ (rc=$rc) — ver $log — SIGO con la próxima"
    fallidas+=("$1")
  else
    echo ">>> [$1]  OK $(date '+%H:%M:%S')"
  fi
  sleep 5   # deja liberar la VRAM antes de la próxima fila
done

echo ""
echo "=================================================================="
echo ">>> BARRIDO COMPLETO $(date '+%H:%M:%S')"
if [ "${#fallidas[@]}" -gt 0 ]; then
  echo ">>> filas FALLIDAS: ${fallidas[*]}"
else
  echo ">>> todas OK"
fi
echo "=================================================================="