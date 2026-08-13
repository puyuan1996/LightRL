#!/usr/bin/env bash
# Stop and remove running tb__*__client containers older than a threshold.

set -Eeuo pipefail

THRESHOLD_HOURS="${THRESHOLD_HOURS:-6}"
STOP_TIMEOUT="${STOP_TIMEOUT:-300}"
DRY_RUN="${DRY_RUN:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

MATCH_PATTERN='^tb__[0-9]+__client([:@].*)?$'

usage() {
    cat <<EOF
用法: $0 [选项]

选项:
  -t, --threshold-hours HOURS  清理运行达到/超过 HOURS 小时的容器，默认: ${THRESHOLD_HOURS}
  -n, --dry-run                只预览，不执行 stop/rm
  -y, --yes                    跳过交互确认，直接执行
      --stop-timeout SECONDS   docker stop 等待秒数，默认: ${STOP_TIMEOUT}
  -h, --help                   显示帮助

环境变量:
  THRESHOLD_HOURS=${THRESHOLD_HOURS} DRY_RUN=${DRY_RUN} ASSUME_YES=${ASSUME_YES} STOP_TIMEOUT=${STOP_TIMEOUT}

安全边界:
  只处理运行中的、镜像名 basename 精确匹配 tb__数字__client 的容器。
EOF
}

die() {
    echo "错误: $*" >&2
    exit 1
}

warn() {
    echo "警告: $*" >&2
}

is_uint() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

image_matches() {
    local image="$1"
    local basename="${image##*/}"
    [[ "$basename" =~ $MATCH_PATTERN ]]
}

parse_epoch() {
    local value="$1"
    local no_fraction

    if date -d "$value" +%s >/dev/null 2>&1; then
        date -d "$value" +%s
        return 0
    fi

    no_fraction="${value%%.*}"
    no_fraction="${no_fraction%Z}"
    if date -j -u -f "%Y-%m-%dT%H:%M:%S" "$no_fraction" +%s >/dev/null 2>&1; then
        date -j -u -f "%Y-%m-%dT%H:%M:%S" "$no_fraction" +%s
        return 0
    fi

    return 1
}

format_duration() {
    local seconds="$1"
    local days hours minutes

    days=$((seconds / 86400))
    hours=$(((seconds % 86400) / 3600))
    minutes=$(((seconds % 3600) / 60))

    if ((days > 0)); then
        printf "%dd%02dh%02dm" "$days" "$hours" "$minutes"
    else
        printf "%dh%02dm" "$hours" "$minutes"
    fi
}

short_id() {
    printf "%.12s" "$1"
}

while (($# > 0)); do
    case "$1" in
        -t|--threshold-hours)
            (($# >= 2)) || die "$1 需要一个小时数"
            THRESHOLD_HOURS="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -y|--yes)
            ASSUME_YES=1
            shift
            ;;
        --stop-timeout)
            (($# >= 2)) || die "$1 需要一个秒数"
            STOP_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "未知参数: $1"
            ;;
    esac
done

is_uint "$THRESHOLD_HOURS" || die "THRESHOLD_HOURS 必须是非负整数"
is_uint "$STOP_TIMEOUT" || die "STOP_TIMEOUT 必须是非负整数"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || die "DRY_RUN 必须是 0 或 1"
[[ "$ASSUME_YES" == "0" || "$ASSUME_YES" == "1" ]] || die "ASSUME_YES 必须是 0 或 1"

command -v "$DOCKER_BIN" >/dev/null 2>&1 || die "找不到 docker 命令: $DOCKER_BIN"

THRESHOLD_SECONDS=$((THRESHOLD_HOURS * 3600))
NOW_EPOCH=$(date +%s)

echo "=========================================="
echo "扫描运行达到/超过 ${THRESHOLD_HOURS} 小时的 tb__*__client 容器"
echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "DRY_RUN = ${DRY_RUN} (1=预览模式, 0=执行模式)"
echo "ASSUME_YES = ${ASSUME_YES} (1=跳过确认, 0=需要确认)"
echo "=========================================="
echo ""

PS_OUTPUT=""
if ! PS_OUTPUT=$("$DOCKER_BIN" ps --no-trunc --format '{{.ID}}\t{{.Image}}\t{{.Names}}\t{{.Status}}'); then
    die "docker ps 执行失败"
fi

TARGET_CIDS=()
TARGET_IMAGES=()
TARGET_NAMES=()
TARGET_RUNTIMES=()
TARGET_STARTED=()
TARGET_STATUS=()

while IFS=$'\t' read -r cid image name status; do
    [[ -n "${cid:-}" ]] || continue
    [[ -n "${image:-}" ]] || continue

    # 这里按镜像字段匹配，避免把 "容器ID 镜像名" 整行拿去匹配导致永远匹配不到。
    if ! image_matches "$image"; then
        continue
    fi

    started_at=""
    if ! started_at=$("$DOCKER_BIN" inspect --format='{{.State.StartedAt}}' "$cid" 2>/dev/null); then
        warn "跳过 $(short_id "$cid")：docker inspect 失败，容器可能已经退出或被删除"
        continue
    fi

    if [[ -z "$started_at" || "$started_at" == "0001-01-01T00:00:00Z" ]]; then
        warn "跳过 $(short_id "$cid")：StartedAt 为空或无效"
        continue
    fi

    if ! start_epoch=$(parse_epoch "$started_at"); then
        warn "跳过 $(short_id "$cid")：无法解析 StartedAt=$started_at"
        continue
    fi

    runtime_seconds=$((NOW_EPOCH - start_epoch))
    if ((runtime_seconds < 0)); then
        warn "跳过 $(short_id "$cid")：启动时间晚于当前时间，可能存在时钟偏差"
        continue
    fi

    if ((runtime_seconds >= THRESHOLD_SECONDS)); then
        TARGET_CIDS+=("$cid")
        TARGET_IMAGES+=("$image")
        TARGET_NAMES+=("${name:-}")
        TARGET_RUNTIMES+=("$(format_duration "$runtime_seconds")")
        TARGET_STARTED+=("$started_at")
        TARGET_STATUS+=("${status:-}")
    fi
done <<< "$PS_OUTPUT"

if ((${#TARGET_CIDS[@]} == 0)); then
    echo "未找到运行达到/超过 ${THRESHOLD_HOURS} 小时的 tb__*__client 容器。"
    exit 0
fi

echo "发现 ${#TARGET_CIDS[@]} 个目标容器："
echo "--------------------------------------------------------------------------------------------------------------"
printf "%-12s %-22s %-32s %-10s %-30s %s\n" "CONTAINER_ID" "IMAGE" "NAME" "RUN_TIME" "STARTED_AT" "STATUS"
echo "--------------------------------------------------------------------------------------------------------------"

for i in "${!TARGET_CIDS[@]}"; do
    printf "%-12s %-22s %-32s %-10s %-30s %s\n" \
        "$(short_id "${TARGET_CIDS[$i]}")" \
        "${TARGET_IMAGES[$i]}" \
        "${TARGET_NAMES[$i]}" \
        "${TARGET_RUNTIMES[$i]}" \
        "${TARGET_STARTED[$i]}" \
        "${TARGET_STATUS[$i]}"
done

echo "--------------------------------------------------------------------------------------------------------------"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] 以上容器将被执行: docker stop -t ${STOP_TIMEOUT} <cid> && docker rm <cid>"
    echo "正式执行可运行: bash $0 --yes"
    exit 0
fi

if [[ "$ASSUME_YES" != "1" ]]; then
    answer=""
    printf "确认停止并删除以上 %d 个容器？输入 yes 继续: " "${#TARGET_CIDS[@]}"
    if ! read -r answer; then
        answer=""
    fi

    if [[ "$answer" != "yes" ]]; then
        echo "已取消，未执行任何清理。"
        exit 0
    fi
fi

echo "开始执行 stop + rm ..."
successes=0
failures=0

for i in "${!TARGET_CIDS[@]}"; do
    cid="${TARGET_CIDS[$i]}"
    image="${TARGET_IMAGES[$i]}"
    runtime="${TARGET_RUNTIMES[$i]}"

    printf "  -> 停止 %s (%s, %s) ... " "$(short_id "$cid")" "$image" "$runtime"
    if "$DOCKER_BIN" stop -t "$STOP_TIMEOUT" "$cid" >/dev/null 2>&1; then
        printf "已停止, "
    else
        running_state=""
        running_state=$("$DOCKER_BIN" inspect --format='{{.State.Running}}' "$cid" 2>/dev/null || true)
        if [[ "$running_state" == "false" ]]; then
            printf "已处于停止状态, "
        else
            echo "停止失败，跳过删除"
            failures=$((failures + 1))
            continue
        fi
    fi

    if "$DOCKER_BIN" rm "$cid" >/dev/null 2>&1; then
        echo "已删除"
        successes=$((successes + 1))
    else
        echo "删除失败"
        failures=$((failures + 1))
    fi
done

echo ""
echo "清理完成：成功 ${successes} 个，失败 ${failures} 个。"

if ((failures > 0)); then
    exit 1
fi
