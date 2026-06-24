#!/bin/bash
# ============================================================
# setup_victim.sh - Persiapan Victim Laptop untuk Testing
# Jalankan di Victim Laptop (yang ada model-nya)
# IP Victim: 192.168.1.10
# ============================================================

EVAL_LOG_DIR="eval_logs"
INTERFACE="eth0"    # Ganti jika nama interface berbeda (cek: ip link show)

echo "============================================================"
echo " SETUP VICTIM LAPTOP - Zeek IDS Testing"
echo "============================================================"

# Cek interface
echo ""
echo "[1] Interface jaringan yang tersedia:"
ip link show | grep -E "^[0-9]+:" | awk '{print "   " $2}'
echo ""
echo "   Interface yang akan digunakan: $INTERFACE"
echo "   Jika salah, edit variabel INTERFACE di script ini."

# Set IP static
echo ""
echo "[2] Setting IP static 192.168.1.10..."
sudo ip addr add 192.168.1.10/24 dev $INTERFACE 2>/dev/null || echo "   (IP mungkin sudah di-set)"
sudo ip link set $INTERFACE up
echo "   IP saat ini:"
ip addr show $INTERFACE | grep "inet " | awk '{print "   " $2}'

# Matikan firewall sementara (agar traffic dari attacker masuk)
echo ""
echo "[3] Menonaktifkan UFW sementara..."
sudo ufw disable 2>/dev/null
echo "   UFW dinonaktifkan."

# Buat direktori log
echo ""
echo "[4] Membuat direktori log: $EVAL_LOG_DIR"
mkdir -p $EVAL_LOG_DIR

# Install dependencies jika belum ada
echo ""
echo "[5] Cek dependencies..."
command -v zeek >/dev/null || echo "   [ERROR] Zeek tidak terinstall!"
command -v python3 >/dev/null && echo "   [OK] Python3" || echo "   [ERROR] Python3 tidak ada"
command -v nc >/dev/null && echo "   [OK] Netcat" || sudo apt install -y netcat-openbsd
python3 -m http.server --help >/dev/null 2>&1 && echo "   [OK] Python HTTP Server"

echo ""
echo "============================================================"
echo " Victim siap. Jalankan Zeek dengan:"
echo "   sudo zeek -C -i $INTERFACE -l $EVAL_LOG_DIR/"
echo ""
echo " Untuk SQLi/Benign test, nyalakan web server dulu:"
echo "   python3 -m http.server 8080 &"
echo ""
echo " Untuk Bruteforce test, nyalakan listener:"
echo "   while true; do nc -l -p 2222; done &"
echo "============================================================"