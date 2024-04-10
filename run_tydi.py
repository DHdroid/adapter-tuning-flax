import os

ADAPTER_NAMES = {
    "20231101.en-fixed-1epoch.pickle": "swahili",
}

COMMAND = """
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


def main(adapter_name, language, is_qry):
    LA_path = f"LA/{adapter_name}"
    TA_path = f"TA/{adapter_name}"
    dataset_name = "castorini/mr-tydi" if is_qry else "castorini/mr-tydi-corpus"
    dataset_split = "test" if is_qry else "train"
    command = COMMAND.format(LA_path=LA_path,
                             TA_path=TA_path,
                             dataset_name=dataset_name,
                             language=language,
                             dataset_split=dataset_split,
                             is_qry="True" if is_qry else "False",
                             encoded_save_path=f"encoded/{adapter_name.split('.')[1]}_{language}_{'query' if is_qry else 'corpus'}.pkl")
    os.system(command)
    return


if __name__ == "__main__":
    for adapter_name, language in ADAPTER_NAMES.items():
        main(adapter_name, language, True)
        main(adapter_name, language, False)


"""
python run_tydi.py

python -m tevatron.faiss_retriever \
--query_reps encoded/en-fixed-1epoch_swahili_query.pkl \
--passage_reps encoded/en-fixed-1epoch_swahili_corpus.pkl \
--depth 100 \
--batch_size -1 \
--save_text \
--save_ranking_to run.mrtydi.swahili.test.txt

python -m tevatron.utils.format.convert_result_to_trec --input run.mrtydi.swahili.test.txt --output run.mrtydi.swahili.test.trec
python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 mrtydi-v1.0-swahili/qrels.test.txt run.mrtydi.swahili.test.trec


wget https://git.uwaterloo.ca/jimmylin/mr.tydi/-/raw/master/data/mrtydi-v1.0-swahili.tar.gz
tar -xvf mrtydi-v1.0-swahili.tar.gz
"""
