#!/usr/bin/env bash
set -euo pipefail

data_dir="$(dirname "$0")"

download_file() {
	output_path="$1"
	url="$2"

	if command -v wget >/dev/null 2>&1; then
		wget -O "$output_path" "$url"
	elif command -v curl >/dev/null 2>&1; then
		curl -L -o "$output_path" "$url"
	else
		echo "Error: neither wget nor curl is installed."
		return 1
	fi
}

download_from_hf() {
	mode="$1"
	echo "Falling back to Hugging Face dataset for mode '$mode'..."
	if [ "$mode" == "episode-data" ]; then
		echo "Error: Hugging Face fallback supports only inter/intra/supervised, not episode-data."
		return 1
	fi
	if command -v uv >/dev/null 2>&1; then
		uv run --with datasets python3 "$data_dir/download_hf.py" --mode "$mode"
		return $?
	fi
	if command -v python3 >/dev/null 2>&1; then
		python3 "$data_dir/download_hf.py" --mode "$mode"
		return $?
	fi
	echo "Error: uv or python3 is required for Hugging Face fallback."
	return 1
}

download_and_extract() {
	mode="$1"
	zip_name="$2"
	url="$3"
	zip_path="$data_dir/$zip_name"

	if ! command -v unzip >/dev/null 2>&1; then
		echo "Error: unzip is not installed."
		return 1
	fi

	if ! download_file "$zip_path" "$url"; then
		download_from_hf "$mode"
		return $?
	fi

	if ! unzip -o -d "$data_dir/" "$zip_path"; then
		rm -f "$zip_path"
		download_from_hf "$mode"
		return $?
	fi

	rm -f "$zip_path"
}

if [ $# == 0 ]; then
	bash "$data_dir/download.sh" inter
	bash "$data_dir/download.sh" intra
	bash "$data_dir/download.sh" supervised
	bash "$data_dir/download.sh" episode-data
elif [ "$1" == "supervised" ]; then
	download_and_extract "supervised" "supervised.zip" "https://cloud.tsinghua.edu.cn/f/c1f71c011d6b461786bc/?dl=1"
elif [ "$1" == "inter" ]; then
	download_and_extract "inter" "inter.zip" "https://cloud.tsinghua.edu.cn/f/3d84d34dc5d845a2bed2/?dl=1"
elif [ "$1" == "intra" ]; then
	download_and_extract "intra" "intra.zip" "https://cloud.tsinghua.edu.cn/f/a176a4870f0a4f8ba0db/?dl=1"
elif [ "$1" == "episode-data" ]; then
	download_and_extract "episode-data" "episode-data.zip" "https://cloud.tsinghua.edu.cn/f/56fb277d3fd2437a8ee3/?dl=1"
else
	echo "Usage: bash $data_dir/download.sh [inter|intra|supervised|episode-data]"
	exit 1
fi

