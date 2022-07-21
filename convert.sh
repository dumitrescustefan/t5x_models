# PATH TO FOLDER WITH CHECKPOINTS
folder=mt5x-base 
SCRIPT=$(realpath -s "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

for file in $folder/* ; do
    last_part="$(cut -d'/' -f2 <<<"$file")"
    python3 convert_flax_to_pytorch.py $SCRIPTPATH/$folder"_fl"/$last_part $SCRIPTPATH/$folder"_pt"/$last_part
    echo "Converted $last_part to Flax ..."
    python3 convert_t5x_checkpoint_to_flax.py --t5x_checkpoint_path $SCRIPTPATH/$folder/$last_part --config_name $SCRIPTPATH/$folder"_fl"/config.json --flax_dump_folder_path $SCRIPTPATH/$folder"_fl"/$last_part
    echo "Converted $last_part to Pytorch ..."
done
