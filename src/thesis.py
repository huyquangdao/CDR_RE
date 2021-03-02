"""This is the first module"""

import sys

from config.cdr_config import CDRConfig
from corpus.cdr_corpus import CDRCorpus
from dataset.cdr_dataset import CDRDataset
from torch.utils.data import DataLoader
from model.cdr_model import GraphEncoder, GraphStateLSTM


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
        ),
        elmo_tensor_dict,
        flair_tensor_dict,
        labels,
    ) = corpus.prepare_features_for_one_dataset(
        config.train_file_path, config.train_elmo_path, config.train_flair_path
    )

    # train_dataset = CDRDataset(
    #     all_doc_token_ids,
    #     all_in_nodes_idx,
    #     all_out_nodes_idx,
    #     all_in_edge_label_ids,
    #     all_out_edge_label_ids,
    #     all_doc_pos_ids,
    #     all_doc_char_ids,
    #     all_doc_hypernym_ids,
    #     all_doc_synonym_ids,
    #     elmo_tensor_dict,
    #     flair_tensor_dict,
    #     all_entity_mapping,
    #     all_ner_labels,
    #     labels,
    # )

    # train_dataset.set_vocabs(
    #     corpus.word_vocab,
    #     corpus.rel_vocab,
    #     corpus.pos_vocab,
    #     corpus.hypernym_vocab,
    #     corpus.synonym_vocab,
    #     corpus.char_vocab,
    # )

    # train_loader = DataLoader(train_dataset, 2, shuffle=True, collate_fn=train_dataset.collate_fn)

    # for batch in train_loader:
    #     for t in batch:
    #         print(t.shape)
    #     break

    encoder = GraphEncoder(
        time_step=config.time_step,
        word_vocab_size=len(corpus.word_vocab),
        egde_vocab_size=len(corpus.rel_vocab),
        pos_vocab_size=len(corpus.pos_vocab),
        char_vocab_size=len(corpus.char_vocab),
        hypernym_vocab_size=len(corpus.hypernym_vocab),
        synonym_vocab_size=len(corpus.synonym_vocab),
        word_embedding_dim=config.word_embedding_dim,
        edge_embedding_dim=config.rel_embedding_dim,
        pos_embedding_dim=config.pos_embedding_dim,
        combined_embedding_dim=config.combined_embedding_dim,
        transformer_attn_head=config.transformer_attn_head,
        transformer_block=config.transformer_block,
        use_transformer=config.use_transformer,
        use_self_atentive=config.use_self_attentive,
        # drop_out=config.drop_out,
        encoder_hidden_size=config.encoder_hidden_size,
    )

    model = GraphStateLSTM(
        relation_classes=config.relation_classes,
        ner_classes=config.ner_classes,
        encoder=encoder,
        glstm_hidden_size=config.glstm_hidden_size,
        elmo_hidden_size=config.elmo_hidden_size,
        flair_hidden_size=config.flair_hidden_size,
        max_distant=config.max_distant,
        distant_embedding_dim=config.distant_embedding_dim,
        ner_hidden_size=config.ner_hidden_size,
        lstm_layers=config.lstm_layers,
        use_ner=config.use_ner,
        # drop_out=config.drop_out,
    )
