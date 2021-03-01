"""This is the first module"""

import sys

from config.cdr_config import CDRConfig
from corpus.cdr_corpus import CDRCorpus
from dataset.cdr_dataset import CDRDataset
from torch.utils.data import DataLoader


def say_hello(text: str) -> None:
    """[summary]

    Args:
        text (str): [the string that you want to print in the screen]

    Returns:
        [str]: [the input string]
    """
    print(text)


if __name__ == "__main__":

    config_file_path = "data/config.json"
    config = CDRConfig.from_json_file(config_file_path)
    corpus = CDRCorpus(config)
    # corpus.prepare_all_vocabs()
    (
        all_doc_token_ids,
        all_in_nodes_idx,
        all_out_nodes_idx,
        all_in_edge_label_ids,
        all_out_edge_label_ids,
        all_doc_pos_ids,
        all_doc_char_ids,
        all_doc_hypernym_ids,
        all_doc_synonym_ids,
        all_entity_mapping,
        all_ner_labels,
    ), elmo_tensor_dict, flair_tensor_dict, labels = corpus.prepare_features_for_one_dataset(
        config.train_file_path, config.train_elmo_path, config.train_flair_path
    )

    train_dataset = CDRDataset(
        all_doc_token_ids,
        all_in_nodes_idx,
        all_out_nodes_idx,
        all_in_edge_label_ids,
        all_out_edge_label_ids,
        all_doc_pos_ids,
        all_doc_char_ids,
        all_doc_hypernym_ids,
        all_doc_synonym_ids,
        elmo_tensor_dict,
        flair_tensor_dict,
        all_entity_mapping,
        all_ner_labels,
        labels,
    )

    train_dataset.set_vocabs(
        corpus.word_vocab, corpus.rel_vocab, corpus.pos_vocab, corpus.hypernym_vocab, corpus.synonym_vocab
    )

    train_loader = DataLoader(train_dataset, 2, shuffle= True, collate_fn=train_dataset.collate_fn)

    from batch in train_loader:
        for t in batch:
            print(t.shape)
        break
    
    
