import os

ADAPTER_NAMES = [
    {
        "LA": "en_prev.pkl",
        "TA": "en_prev.pkl",
        "lang": "telugu"
    }
]

DATA_PREPARATION_COMMAND = """
wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-{language}.tar.gz
tar -xvf mrtydi-v1.0-{language}.tar.gz
mv mrtydi-v1.0-{language} ./mrtydi
rm mrtydi-v1.0-{language}.tar.gz
"""

ENCODE_COMMAND = """
python -u encode_dataset.py \
    dataset.dataset_name={dataset_name} \
    dataset.dataset_language={language} \
    dataset.dataset_split={dataset_split} \
    dataset.q_max_len=16 \
    dataset.p_max_len=128 \
    dataset.encode_is_qry={is_qry} \
    dataset.encoded_save_path={encoded_save_path} \
    model.adapters.0.pretrained_weights={LA_path} \
    model.adapters.1.pretrained_weights={TA_path} \
"""

FAISS_COMMAND = """
python -m tevatron.faiss_retriever \
    --query_reps {query_path} \
    --passage_reps {passage_path} \
    --depth 100 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to eval/run.mrtydi.{language}.test.txt
"""

EVAL_COMMAND = """
python -m tevatron.utils.format.convert_result_to_trec --input eval/run.mrtydi.{language}.test.txt --output eval/run.mrtydi.{language}.test.trec
python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 mrtydi/mrtydi-v1.0-{language}/qrels.test.txt eval/run.mrtydi.{language}.test.trec
"""

def main(la_name, ta_name, language, is_qry):
    LA_path = f"LA/{la_name}"
    TA_path = f"TA/{ta_name}"
    dataset_name = "castorini/mr-tydi" if is_qry else "castorini/mr-tydi-corpus"
    dataset_split = "test" if is_qry else "train"
    encoded_path = f"encoded/{la_name.split('.')[0]}_{ta_name.split('.')[0]}_{language}_{'query' if is_qry else 'corpus'}.pkl"
    command = ENCODE_COMMAND.format(LA_path=LA_path,
                             TA_path=TA_path,
                             dataset_name=dataset_name,
                             language=language,
                             dataset_split=dataset_split,
                             is_qry="True" if is_qry else "False",
                             encoded_save_path=encoded_path)
    os.system(command)
    return encoded_path


if __name__ == "__main__":
    for exp_dict in ADAPTER_NAMES:
        if not os.path.exists(f"mrtydi/mrtydi-v1.0-{exp_dict['lang']}"):
            os.system(DATA_PREPARATION_COMMAND.format(language=exp_dict['lang']))
        qry_path = main(exp_dict["LA"], exp_dict["TA"], exp_dict["lang"], True)
        passage_path = main(exp_dict["LA"], exp_dict["TA"], exp_dict["lang"], False)
        os.system(FAISS_COMMAND.format(query_path=qry_path, passage_path=passage_path, language=exp_dict["lang"]))
        os.system(EVAL_COMMAND.format(language=exp_dict["lang"]))



"""
python run_tydi.py

python -m tevatron.faiss_retriever \
--query_reps encoded/sw_4_en_prev_swahili_query.pkl \
--passage_reps encoded/sw_4_en_prev_swahili_corpus.pkl \
--depth 100 \
--batch_size -1 \
--save_text \
--save_ranking_to run.mrtydi.swahili.test.txt

python -m tevatron.utils.format.convert_result_to_trec --input run.mrtydi.swahili.test.txt --output run.mrtydi.swahili.test.trec
python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 mrtydi-v1.0-swahili/qrels.test.txt run.mrtydi.swahili.test.trec


wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-swahili.tar.gz
tar -xvf mrtydi-v1.0-swahili.tar.gz
"""
