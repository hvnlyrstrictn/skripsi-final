#!/bin/bash
# mass_zeek.sh - Per-File Processing (Robust Version)
# Proses setiap PCAP secara terpisah, gabungkan log hasilnya

ESSENTIAL_LOGS=("conn" "http" "dns" "files" "weird" "ssl")

echo "============================================================"
echo " MULAI PROSES MASSAL ZEEK (PER-FILE MODE)"
echo "============================================================"

TOTAL_SUCCESS=0

for dir in */; do
    [ -d "$dir" ] || continue
    cd "$dir"
    folder_name="${dir%/}"
    echo ""
    echo "[FOLDER] Processing: $folder_name"

    # Temukan semua file pcap/pcapng
    mapfile -t pcap_files < <(find . -maxdepth 1 \( -name "*.pcap" -o -name "*.pcapng" \) | sort)
    count="${#pcap_files[@]}"

    if [ "$count" -eq 0 ]; then
        echo "  -> SKIP: Tidak ada file PCAP."
        cd ..
        continue
    fi

    echo "  -> Ditemukan $count file: ${pcap_files[*]}"

    # Bersihkan log lama
    rm -f *.log temp_work_dir 2>/dev/null
    mkdir -p temp_work_dir

    # Proses setiap file PCAP secara terpisah
    for pcap_file in "${pcap_files[@]}"; do
        pcap_basename=$(basename "$pcap_file" .pcap)
        pcap_basename=$(basename "$pcap_basename" .pcapng)
        work_dir="temp_work_dir/${pcap_basename}"
        mkdir -p "$work_dir"

        echo "  -> Zeek: $pcap_file"
        (cd "$work_dir" && zeek -C -r "../../$pcap_file" 2>/dev/null)
    done

    # Gabungkan log dari semua subfolder
    echo "  -> Menggabungkan log..."
    saved_count=0

    for log_type in "${ESSENTIAL_LOGS[@]}"; do
        combined_file="${log_type}_${folder_name}.log"
        header_written=false

        for work_subdir in temp_work_dir/*/; do
            src_log="${work_subdir}${log_type}.log"
            if [ -f "$src_log" ]; then
                if [ "$header_written" = false ]; then
                    # Tulis header dari file pertama
                    grep "^#" "$src_log" > "$combined_file" 2>/dev/null
                    header_written=true
                fi
                # Append hanya baris data (bukan header)
                grep -v "^#" "$src_log" >> "$combined_file" 2>/dev/null
            fi
        done

        if [ -f "$combined_file" ] && [ -s "$combined_file" ]; then
            line_count=$(grep -v "^#" "$combined_file" | wc -l)
            echo "     [OK] $combined_file ($line_count baris data)"
            saved_count=$((saved_count + 1))
        else
            rm -f "$combined_file" 2>/dev/null
        fi
    done

    # Bersihkan folder kerja sementara
    rm -rf temp_work_dir

    if [ "$saved_count" -gt 0 ]; then
        echo "  -> SUKSES: $saved_count log file(s) disimpan."
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    else
        echo "  -> WARNING: Tidak ada log yang dihasilkan."
    fi

    cd ..
done

echo ""
echo "============================================================"
echo " SELESAI - $TOTAL_SUCCESS folder berhasil diproses"
echo "============================================================"
echo ""
echo "Semua log yang dihasilkan:"
find . -name "*_0*.log" | sort