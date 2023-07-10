MODEL_TYPE=$1
PURPOSE=$2

for i in 1 2 3
do
  python3 trainer.py --conv_type ${i} --purpose $PURPOSE --model_type $MODEL_TYPE
  python3 infer.py --conv_type ${i} --purpose $PURPOSE --model_type $MODEL_TYPE
  mv infer.csv infer_${i}.csv
done

python3 merge.py --purpose $PURPOSE

if [ "$PURPOSE" = "validation" ]
then
  python3 cal_auc.py
fi

rm -rf infer_*.csv
rm -rf ./output_t*
