echo "Start Training"

epoch_lo=1
epoch_up=5
batch_lo=10
batch_up=20
results=()
max_value=-1
best_epoch=1
best_batch=10

for ((param1=$epoch_lo; param1<=$epoch_up; param1++)); do
    for ((param2=$batch_lo; param2<=$batch_up; param2++)); do
        
        # 运行 Python 模型并传递参数
        #echo "$param1"
        acc=$(python Q12.py "$param1" "$param2")
        results+=("$acc")
        if (( acc > max_value )); then
            max_value=$acc
            best_epoch="$param1"
            best_batch="$param2"
        fi

    done
done

echo "best acc: $max_value, corresponding epoch: $best_epoch, corresponding batch: $best_batch"
