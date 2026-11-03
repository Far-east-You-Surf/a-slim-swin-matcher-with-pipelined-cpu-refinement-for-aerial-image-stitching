#!/usr/bin/env bash
# P21 ulang (2026-10-15 malam): satu pod per pemanggilan, dari buat -> unggah -> jalankan -> tarik hasil BERKALA ->
# verifikasi -> terminate. Dibuat karena pod L4/A100 siang tadi ter-terminate sebelum hasilnya disalin.
# Pakai (dari cloud_e14/): bash pod4/run_pod_A.sh <tag> "<gpu-id>" <batas-jam>   (Secure saja; tanpa percobaan ulang)
#   contoh: bash pod4/run_pod_A.sh l4 "NVIDIA L4" 5
#   POD_DIR=pod4_edm bash pod4/run_pod_A.sh l4_edm "NVIDIA L4" 1   -> hanya ONNX di pod4_edm/onnx (jumlah harapan
#   latensi = 3 x ONNX, energi = 2 x ONNX + 3 idle; untuk pod4 = 87 dan 61)
# 2026-11-02: daftar MD5 diurutkan LC_ALL=C di kedua sisi (sesi 00:20 & 00:42 gagal palsu karena urutan glob beda).
# Hasil: results/B_<tag>/ (results pod ditarik tiap 5 menit, jadi data parsial pun tersimpan bila pod hilang).
# Terminate otomatis hanya bila (a) sesiA.done + 42 latensi + 31 energi lengkap, atau (b) batas jam terlampaui
# (hasil parsial ditarik dulu). Tanpa API key di berkas: runpodctl membaca ~/.runpod sendiri.
set -u
cd "$(dirname "$0")/.."
TAG=$1; GPU=$2; MAXH=$3
OUT=results/A_$TAG; mkdir -p "$OUT"
PODDIR=${POD_DIR:-pod4}
NONNX=$(ls "$PODDIR"/onnx/*.onnx | wc -l); NL_EXP=$((3 * NONNX)); NE_EXP=$((2 * NONNX + 3))
LOG=$OUT/driver.log
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
KEY=~/.runpod/ssh/runpodctl-ssh-key
PUB="$(cat ~/.runpod/ssh/runpodctl-ssh-key.pub)"
START='apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && mkdir -p /root/.ssh /run/sshd && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd && sleep infinity'
ENVJ=$(python3 -c 'import json,sys;print(json.dumps({"PUBLIC_KEY":sys.argv[1]}))' "$PUB")

if [ -n "${ATTACH_ID:-}" ]; then
  # 2026-11-02 21:10: driver sesi l4_edm mati bersama listrik (20:51) saat menunggu ssh, pod tetap hidup tanpa
  # pengawas. ATTACH_ID=<id> menyambung ke pod itu (tanpa membuat pod baru), lalu langkah yang sama dijalankan.
  log "menyambung ke pod $ATTACH_ID yang sudah ada ($TAG, $GPU)"
  [ -f "$OUT/pod_create.json" ] && cp "$OUT/pod_create.json" "$OUT/pod_create_awal.json"
  runpodctl pod get "$ATTACH_ID" > "$OUT/pod_create.json" 2>&1
else
log "membuat pod $TAG ($GPU)"
runpodctl pod create --name "arevA-$TAG" --gpu-id "$GPU" --cloud-type SECURE --image nvcr.io/nvidia/tensorrt:26.07-py3 \
  --container-disk-in-gb 50 --ports 22/tcp --env "$ENVJ" --docker-args "bash -c '$START'" \
  --wait --wait-timeout 15m > "$OUT/pod_create.json" 2>&1
fi
ID=$(python3 - "$OUT/pod_create.json" <<'EOF'
import json,sys,re
t=open(sys.argv[1]).read(); i=t.find('{')
try: print(json.loads(t[i:])['id'])
except Exception:
    m=re.search(r'pod ([a-z0-9]{10,})',t); print(m.group(1) if m else '')
EOF
)
[ -n "$ID" ] || { log "GAGAL membuat pod; lihat pod_create.json"; exit 1; }
echo "$ID" > "$OUT/pod_id.txt"; date -Is > "$OUT/pod_start_time.txt"
if [ -n "${ATTACH_ID:-}" ]; then   # biaya dihitung dari saat pod benar-benar disewa (createdAt), bukan saat menyambung
  python3 -c 'import json,sys,datetime as D;d=json.load(open(sys.argv[1]));t=D.datetime.strptime(d["createdAt"][:19],"%Y-%m-%d %H:%M:%S").replace(tzinfo=D.timezone.utc);print(t.astimezone().isoformat(timespec="seconds"))' "$OUT/pod_create.json" > "$OUT/pod_start_time.txt"
fi
read -r IP PORT < <(python3 -c 'import json,sys;t=open(sys.argv[1]).read();d=json.loads(t[t.find("{"):]);print(d["ssh"]["ip"],d["ssh"]["port"])' "$OUT/pod_create.json" 2>/dev/null)
[ -n "${IP:-}" ] || { log "pod $ID dibuat tetapi ssh tidak siap -> terminate"; runpodctl pod delete "$ID" >>"$LOG" 2>&1; exit 1; }
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=20 root@$IP"
log "pod $ID siap di $IP:$PORT"

for i in $(seq 1 20); do $SSH true 2>/dev/null && break; sleep 15; done
log "unggah onnx/ + scripts/ (symlink diikuti)"
(cd "$PODDIR" && tar -chf - onnx scripts) | $SSH 'mkdir -p /workspace && tar -xf - -C /workspace' || { log "unggah GAGAL -> terminate"; runpodctl pod delete "$ID" >>"$LOG" 2>&1; exit 1; }
(cd "$PODDIR" && for f in onnx/*.onnx; do echo "$(md5sum < "$f" | cut -d' ' -f1)  $f"; done) | LC_ALL=C sort -k2 > "$OUT/md5_local.txt"
$SSH 'cd /workspace && for f in onnx/*.onnx; do echo "$(md5sum < $f | cut -d" " -f1)  $f"; done' | LC_ALL=C sort -k2 > "$OUT/md5_pod.txt"
if ! diff -q "$OUT/md5_local.txt" "$OUT/md5_pod.txt" >/dev/null; then log "MD5 tidak cocok -> terminate"; runpodctl pod delete "$ID" >>"$LOG" 2>&1; exit 1; fi
log "MD5 cocok ($(wc -l < "$OUT/md5_pod.txt") ONNX); memulai pod_session_A.sh"
# Sesi 20:05 tertahan di sini (ssh menunggu); -f membuat klien ssh ke latar segera setelah perintah dikirim.
timeout 60 $SSH -f -n 'cd /workspace && setsid nohup bash scripts/pod_session_A.sh > /workspace/session.out 2>&1 < /dev/null &' < /dev/null

pull(){ $SSH 'cd /workspace && tar -cf - results session.out' 2>/dev/null | tar -xf - -C "$OUT" 2>/dev/null; }
count(){ local n=0; for s in 1 2 3; do n=$((n+$(ls "$OUT"/results/latA-$s/latency/*.json 2>/dev/null | wc -l))); done
         echo "$n $(ls "$OUT"/results/energiA/energy/*.json 2>/dev/null | wc -l)"; }
T0=$(date +%s)
while :; do
  sleep 300
  pull
  read -r NL NE < <(count)
  DONE=0; [ -f "$OUT/results/sesiA.done" ] && DONE=1
  log "tarik: latensi $NL/$NL_EXP, energi $NE/$NE_EXP, done=$DONE"
  if [ $DONE = 1 ]; then
    pull; read -r NL NE < <(count)
    ERR=$(grep -ciE 'traceback|error' "$OUT"/results/*.log 2>/dev/null | awk -F: '{s+=$2}END{print s+0}')
    if [ "$NL" = "$NL_EXP" ] && [ "$NE" = "$NE_EXP" ]; then
      log "LENGKAP (latensi $NL, energi $NE, baris error di log: $ERR) -> terminate $ID"; break
    fi
    log "done tetapi TIDAK lengkap (latensi $NL, energi $NE, error $ERR) -> hasil ditarik, terminate"; break
  fi
  if [ $(( $(date +%s) - T0 )) -gt $(( MAXH*3600 )) ]; then log "batas $MAXH jam terlampaui -> tarik parsial, terminate"; pull; break; fi
  if ! runpodctl pod get "$ID" >/dev/null 2>&1; then log "POD HILANG dari akun (hasil terakhir sudah ditarik)"; exit 2; fi
done
runpodctl pod delete "$ID" >>"$LOG" 2>&1
sleep 20
if runpodctl pod get "$ID" >/dev/null 2>&1; then log "PERINGATAN: pod $ID masih ada sesudah delete"; else log "pod $ID ter-terminate"; fi
touch "$OUT/driver.done"
