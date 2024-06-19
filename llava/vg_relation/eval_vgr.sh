
filename="llava_vgr_QCM_yesno"
python eval_vgr_yesno.py \
--result-file ~/llava_probing/vgr/results/base_model/${filename}.json \
--data-file ~/VG_Relation/${filename}.json \
--output-file ~/llava_probing/vgr/results/base_model/${filename}_output.json \
--output-result   ~/llava_probing/vgr/results/base_model/${filename}_result.json >>  ~/llava_probing/vgr/results/base_model_results/${filename}.txt 