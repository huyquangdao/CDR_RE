"""This module describe how we prepare the training, development and testing dataset from Biocreative CDR5 corpus."""
import codecs
import itertools
import json
import os
import pickle
import random
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")
import spacy

from dataset.cdr_dataset import CDRDataset

SOME_SPECIFIC_MENTIONS = [
    "tyrosine-methionine-aspartate-aspartate",
    "anemia/thrombocytopenia/emesis/rash",
    "metoprolol/alpha-hydroxymetoprolol",
    "glutamate/N-methyl-D-aspartate",
    "cyclosporine-and-prednisone-treated",
    "platinum/paclitaxel-refractory",
]


ner_vocab = {"O": 0, "B_Chemical": 1, "I_Chemical": 2, "B_Disease": 3, "I_Disease": 4}
ner_idx2label = {0: "O", 1: "B_Chemical", 2: "I_Chemical", 3: "B_Disease", 4: "I_Disease"}
# idx2word = {k: v for v, k in word_vocab.items()}


LabelAnnotationType = Dict[Tuple[Any, Any, Any], Any]
DocAnnotationType = Dict[Any, List[Union[List[List[Any]], Any, Any]]]

nlp = spacy.load("en_core_sci_md")


class CDRCorpus:

    """[summary]"""

    def __init__(self, config):
        """[summary]

        Args:
            config ([type]): [description]
        """
        self.config = config

    def load_vocab(self, file_path):
        with open(file_path) as f:
            vocab = json.load(f)
            return vocab

    def save_vocab(self, vocab, file_path):
        with open(file_path, "w") as f:
            json.dump(vocab, f)

    def load_tensor(self, tensor_file_path):
        with open(tensor_file_path, "rb") as f:
            tensor = pickle.load(f)
            return tensor
    
    def load_numpy(self, numpy_file_path):
        with open(numpy_file_path, 'rb') as f:
            matrix = np.load(f)
            return matrix

    def prepare_features_for_one_dataset(self, data_file_path, elmo_file_path, flair_file_path):
        in_adjacency_dict, out_adjacency_dict, entity_mapping_dict, labels = self.process_dataset(data_file_path)
        if os.path.exists(self.config.word_vocab_path):
            self.word_vocab = self.load_vocab(self.config.word_vocab_path)
            self.rel_vocab = self.load_vocab(self.config.rel_vocab_path)
            self.pos_vocab = self.load_vocab(self.config.pos_vocab_path)
            self.hypernym_vocab = self.load_vocab(self.config.hypernym_vocab_path)
            self.synonym_vocab = self.load_vocab(self.config.synonym_vocab_path)
            self.char_vocab = self.load_vocab(self.config.char_vocab_path)
        else:
            raise Exception(
                "You need prepare whole datasets to creative all vocab files. please use prepare_all_vocabs method"
            )

        elmo_tensor = self.load_tensor(elmo_file_path)
        flair_tensor = self.load_tensor(flair_file_path)

        return (
            self.convert_examples_to_features(
                in_adjacency_dict,
                out_adjacency_dict,
                entity_mapping_dict,
                self.word_vocab,
                self.rel_vocab,
                self.pos_vocab,
                self.char_vocab,
                self.hypernym_vocab,
                self.synonym_vocab,
            ),
            elmo_tensor,
            flair_tensor,
            labels,
        )

    def prepare_all_vocabs(self) -> None:
        """[summary]

        Returns:
            [type]: [description]
        """
        (
            train_in_adjacency_dict,
            train_out_adjacency_dict,
            train_entity_mapping_dict,
            train_all_labels,
        ) = self.process_dataset(self.config.train_file_path)
        dev_in_adjacency_dict, dev_out_adjacency_dict, dev_entity_mapping_dict, dev_all_labels = self.process_dataset(
            self.config.dev_file_path
        )
        (
            test_in_adjacency_dict,
            test_out_adjacency_dict,
            test_entity_mapping_dict,
            test_all_labels,
        ) = self.process_dataset(self.config.test_file_path)

        if not os.path.exists(self.config.word_vocab_path):
            (
                word_vocab,
                rel_vocab,
                pos_vocab,
                char_vocab,
                hypernym_vocab,
                synonym_vocab,
            ) = self.create_vocabs([train_in_adjacency_dict, dev_in_adjacency_dict, test_in_adjacency_dict])
            self.save_vocab(word_vocab, self.config.word_vocab_path)
            self.save_vocab(rel_vocab, self.config.rel_vocab_path)
            self.save_vocab(pos_vocab, self.config.pos_vocab_path)
            self.save_vocab(char_vocab, self.config.char_vocab_path)
            self.save_vocab(hypernym_vocab, self.config.hypernym_vocab_path)
            self.save_vocab(synonym_vocab, self.config.synonym_vocab_path)
        else:
            self.word_vocab = self.load_vocab(self.config.word_vocab_path)
            self.rel_vocab = self.load_vocab(self.config.rel_vocab_path)
            self.pos_vocab = self.load_vocab(self.config.pos_vocab_path)
            self.hypernym_vocab = self.load_vocab(self.config.hypernym_vocab_path)
            self.synonym_vocab = self.load_vocab(self.config.synonym_vocab_path)

    def make_pairs(self, entity_annotations: List[Tuple[Any, Any, Any, Any, Any]]) -> List[Tuple[Any, Any]]:
        """[summary]

        Args:
            entity_annotations (List[Tuple[Any, Any, Any, Any, Any]]): [description]

        Returns:
            List[Tuple[Any, Any]]: [description]
        """
        chem_entity_ids = [anno[-1] for anno in entity_annotations if anno[-2] == self.config.chemical_string]
        dis_entity_ids = [anno[-1] for anno in entity_annotations if anno[-2] == self.config.disease_string]

        chem_entity_ids = list(set(chem_entity_ids))
        dis_entity_ids = list(set(dis_entity_ids))

        chem_dis_pair_ids = list(itertools.product(chem_entity_ids, dis_entity_ids))

        return chem_dis_pair_ids

    def get_valid_entity_mentions(
        self, entity_mentions_annotations: List[Tuple[Any, Any, Any, Any, Any]], invalid_id: str = "-1"
    ) -> List[Tuple[Any, Any, Any, Any, Any]]:
        """Remove all entity which has unknown id.

        Args:
            entity_mentions_annotations (List[Tuple[Any, Any, Any, Any, Any]]): list of entity mention annotations,
            whose each element is a tuple of (start_offset, end_offset, text, entity type, mesh_id).
            invalid_id (int, optional): The unknown entity id from CDR5. Defaults to '-1'.

        Returns:
            [type]: [description]
        """

        # remove entity anno in document's title and entity with id = -1
        return [mention_anno for mention_anno in entity_mentions_annotations if mention_anno[-1] != invalid_id]

    def remove_entity_mention_in_title(
        self, entity_mentions_annotations: List[Tuple[Any, Any, Any, Any, Any]], title
    ) -> List[Tuple[Any, Any, Any, Any, Any]]:
        """[summary]

        Args:
            entity_mentions_annotations (List[Tuple[Any, Any, Any, Any, Any]]): [description]
            title ([type]): [description]

        Returns:
            List[Tuple[Any, Any, Any, Any, Any]]: [description]
        """
        return [mention_anno for mention_anno in entity_mentions_annotations if int(mention_anno[1]) >= len(title)]

    def read_raw_dataset(self, file_path: str) -> Tuple[LabelAnnotationType, DocAnnotationType]:
        """Read the raw biocreative CDR5 dataset

        Args:
            file_path (str): path to the dataset

        Returns:
            Tuple[Label_annotation_type, Doc_annotation_type]: A tuple of two dictionary, the label
            annotation whose each key is a tuple of (chemical_mesh_id, disease_mesh_id, document_id)
            and its value is the relation (eg: CID or None)
            the document annotation contains key, values pairs with key is the document id
            and value is a list whose elements are the document title, abstract and
            list of entity mention annotations respectively.
        """

        with open(file_path) as f_raw:
            lines = f_raw.read().split("\n")

            raw_doc_annotations = [list(group) for k, group in groupby(lines, lambda x: x == "") if not k]
            label_annotations = {}
            doc_annotations = {}

            for doc_annos in raw_doc_annotations:

                title = None
                abstract = None
                current_annotations = []

                for anno in doc_annos:

                    if "|t|" in anno:
                        pud_id, title = anno.strip().split("|t|")
                    elif "|a|" in anno:
                        pub_id, abstract = anno.strip().split("|a|")
                    else:
                        splits = anno.strip().split("\t")
                        if len(splits) == 4:
                            _, rel, e1_id, e2_id = splits
                            label_annotations[(e1_id, e2_id, pud_id)] = rel
                        elif len(splits) == 6:
                            _, start, end, mention, label, kg_ids = splits
                            for kg_id in kg_ids.split("|"):
                                current_annotations.append([int(start), int(end), mention, label, kg_id])
                        elif len(splits) == 7:
                            _, start, end, mention, label, kg_ids, split_mentions = splits
                            for kg_id in kg_ids.split("|"):
                                current_annotations.append([int(start), int(end), mention, label, kg_id])

                assert title is not None and abstract is not None

                doc_annotations[pud_id] = [title, abstract, current_annotations]

            return label_annotations, doc_annotations

    def create_sentence_adjacency_dict(self, sent, debug=True):
        """[summary]

        Args:
            sent ([type]): [description]
            debug (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        in_sent_adjacency_dict = {}
        out_sent_adjacency_dict = {}

        root = sent.root

        assert root is not None

        if debug == True:
            svg = displacy.render(sent, style="dep", jupyter=True, options={"collapse_punct": False})

        # dependency rel
        for token in sent:
            out_sent_adjacency_dict[token] = []
            # out adjacency dict of this token appends all the token's childrens and their dependency relation
            for child in token.children:
                out_sent_adjacency_dict[token].append((token, child.dep_))
            # in adjacency dict of this tokens append itself, its head and its dependency relation
            if token != root:
                # in_sent_adjacency_dict[token] = [(token,SELF_REL),(token.head, token.dep_)]
                in_sent_adjacency_dict[token] = [(token.head, token.dep_)]
            else:
                # in_sent_adjacency_dict[token] = [(token,SELF_REL)]
                in_sent_adjacency_dict[token] = []

        return in_sent_adjacency_dict, out_sent_adjacency_dict, root

    def create_document_adjacency_dict(
        self, doc, list_in_sent_adjacency_dict, list_out_sent_adjacency_dict, list_root
    ):
        """[summary]

        Args:
            doc ([type]): [description]
            list_in_sent_adjacency_dict ([type]): [description]
            list_out_sent_adjacency_dict ([type]): [description]
            list_root ([type]): [description]

        Returns:
            [type]: [description]
        """
        in_doc_adjacency_dict = {}
        out_doc_adjacency_dict = {}
        # create in and out adjacency dict for document

        for token in doc:
            for in_sent_adjacency_dict in list_in_sent_adjacency_dict:
                if token in in_sent_adjacency_dict:
                    # the token in current sentence
                    in_doc_adjacency_dict[token] = in_sent_adjacency_dict[token]

            for out_sent_adjacency_dict in list_out_sent_adjacency_dict:
                if token in out_sent_adjacency_dict:
                    # the token in current sentence
                    out_doc_adjacency_dict[token] = out_sent_adjacency_dict[token]

        # extra rel between adjacency tokens
        list_tokens = list(doc)

        for i in range(1, len(list_tokens) - 1):
            in_doc_adjacency_dict[list_tokens[i]].append((list_tokens[i - 1], self.config.adjacency_rel))
            out_doc_adjacency_dict[list_tokens[i]].append((list_tokens[i], self.config.adjacency_rel))

        in_doc_adjacency_dict[list_tokens[-1]].append((list_tokens[-2], self.config.adjacency_rel))
        out_doc_adjacency_dict[list_tokens[0]].append((list_tokens[0], self.config.adjacency_rel))

        # connect root of adjacency sentences, i.e document with more than 2 sentences

        if len(list_root) >= 2:
            for i in range(1, len(list_root) - 1):
                in_doc_adjacency_dict[list_root[i]].append((list_root[i - 1], self.config.root_rel))
                out_doc_adjacency_dict[list_root[i]].append((list_root[i], self.config.root_rel))

            in_doc_adjacency_dict[list_root[-1]].append((list_root[-2], self.config.root_rel))
            out_doc_adjacency_dict[list_root[0]].append((list_root[0], self.config.root_rel))

        return in_doc_adjacency_dict, out_doc_adjacency_dict

    def create_features_one_doc(self, pud_id, abstract, entity_annotations, offset_span=20, debug=False):
        """[summary]

        Args:
            pud_id ([type]): [description]
            abstract ([type]): [description]
            entity_annotations ([type]): [description]
            offset_span (int, optional): [description]. Defaults to 20.
            debug (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        # sentence tokenize
        doc = nlp(abstract)

        list_in_sentence_adjacency_dicts = []
        list_out_sentence_adjacency_dicts = []
        list_roots = []
        for sent in doc.sents:

            # create adjacency list for sentence and find the root of the sentence.
            in_sent_adjacency_dict, out_sent_adjacency_dict, sent_root = self.create_sentence_adjacency_dict(
                sent, debug=debug
            )
            list_in_sentence_adjacency_dicts.append(in_sent_adjacency_dict)
            list_out_sentence_adjacency_dicts.append(out_sent_adjacency_dict)
            list_roots.append(sent_root)

        # merge all sentence denpendency to create document dependency tree.
        in_doc_adjacency_dict, out_doc_adjacency_dict = self.create_document_adjacency_dict(
            doc, list_in_sentence_adjacency_dicts, list_out_sentence_adjacency_dicts, list_roots
        )

        # mapping entity spans to document ids
        entity_mapping = {}

        for en_anno in entity_annotations:

            start, end, mention, label, kg_id = en_anno
            key = (start, end, mention, label, kg_id)
            entity_mapping[key] = []

            for token, adjacen in in_doc_adjacency_dict.items():
                token_start = token.idx
                token_end = token_start + len(token)
                if token_start >= start and token_end <= end:
                    entity_mapping[key].append(token)
                # some annotations which form abc-#mention-abcxyz, so we extra token spans to a pre-defined threshhold.
                elif (token_start >= start - offset_span and token_end <= end + offset_span) and mention in token.text:
                    entity_mapping[key].append(token)
                # hard code for some specific mention
                elif token.text in SOME_SPECIFIC_MENTIONS:
                    entity_mapping[key].append(token)
            try:
                assert entity_mapping[key] != []
            except:
                print(en_anno)
                print(pud_id)
                print(abstract[start - 50 : end + 50])

        return in_doc_adjacency_dict, out_doc_adjacency_dict, entity_mapping

    def preprocess_one_doc(self, pud_id, title, abstract, entity_annotations, debug=False):
        """[summary]

        Args:
            pud_id ([type]): [description]
            title ([type]): [description]
            abstract ([type]): [description]
            entity_annotations ([type]): [description]
            debug (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        # remove all annotations of invalid enity (i.e entity id equals -1)
        entity_annotations = self.get_valid_entity_mentions(entity_annotations)

        if not self.config.use_title:
            # subtract title offset plus one space
            for en_anno in entity_annotations:
                en_anno[0] -= len(title) + 1
                en_anno[1] -= len(title) + 1
            # remove entity mention in the title
            entity_annotations = self.remove_entity_mention_in_title(entity_annotations)

        # make all pairs chemical disease entities
        chem_dis_pair_ids = self.make_pairs(entity_annotations)
        # create doc_adjacency_dict and entity_to_tokens_mapping

        in_doc_adjacency_dict, out_doc_adjacency_dict, entity_mapping = self.create_features_one_doc(
            pud_id, title + " " + abstract if self.config.use_title else abstract, entity_annotations
        )

        # print(chem_dis_pair_ids)
        return chem_dis_pair_ids, in_doc_adjacency_dict, out_doc_adjacency_dict, entity_mapping

    def process_dataset(self, file_path, mesh_filtering=True):
        """[summary]

        Args:
            file_path ([type]): [description]
            mesh_path ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        label_annotations, doc_annotations = self.read_raw_dataset(file_path)

        label_docs = defaultdict(list)
        in_adjacency_dict = {}
        entity_mapping_dict = {}
        out_adjacency_dict = {}

        max_doc_length = -1

        # process all document
        for pud_id, doc_anno in doc_annotations.items():
            title, abstract, entity_annotations = doc_anno
            chem_dis_pair_ids, in_doc_adjacency_dict, out_doc_adjacency_dict, entity_mapping = self.preprocess_one_doc(
                pud_id, title, abstract, entity_annotations
            )
            label_docs[pud_id] = chem_dis_pair_ids
            in_adjacency_dict[pud_id] = in_doc_adjacency_dict
            out_adjacency_dict[pud_id] = out_doc_adjacency_dict
            entity_mapping_dict[pud_id] = entity_mapping

            max_doc_length = max(len(in_doc_adjacency_dict), max_doc_length)

        # gather positive examples and negative examples
        pos_doc_examples = defaultdict(list)
        neg_doc_examples = defaultdict(list)

        unfilterd_positive_count = 0
        unfilterd_negative_count = 0

        for pud_id in doc_annotations.keys():
            for c_e, d_e in label_docs[pud_id]:
                if (c_e, d_e, pud_id) in label_annotations:
                    pos_doc_examples[pud_id].append((c_e, d_e))
                    unfilterd_positive_count += 1
                else:
                    neg_doc_examples[pud_id].append((c_e, d_e))
                    unfilterd_negative_count += 1

        print("original number of positive samples: ", unfilterd_positive_count)
        print("original number of negative samples: ", unfilterd_negative_count)
        print("max document length: ", max_doc_length)

        if self.config.mesh_filtering:
            ent_tree_map = defaultdict(list)
            with codecs.open(self.config.mesh_path, "r", encoding="utf-16-le") as f:
                lines = [l.rstrip().split("\t") for i, l in enumerate(f) if i > 0]
                [ent_tree_map[l[1]].append(l[0]) for l in lines]
                neg_doc_examples, n_filterd_samples = self.filter_with_mesh_vocab(
                    ent_tree_map, pos_doc_examples, neg_doc_examples
                )

            print("number of negative examples are filterd:", n_filterd_samples)

        all_labels = []

        for pud_id, value in pos_doc_examples.items():
            for c_id, d_id in value:
                key = (pud_id, c_id, d_id, "CID")
                all_labels.append(key)

        for pud_id, value in neg_doc_examples.items():
            for c_id, d_id in value:
                key = (pud_id, c_id, d_id, "NULL")
                all_labels.append(key)

        random.shuffle(all_labels)
        print("total samples: ", len(all_labels))
        return in_adjacency_dict, out_adjacency_dict, entity_mapping_dict, all_labels

    def filter_with_mesh_vocab(self, mesh_tree, pos_doc_examples, neg_doc_examples):
        """[summary]

        Args:
            mesh_tree ([type]): [description]
            pos_doc_examples ([type]): [description]
            neg_doc_examples ([type]): [description]

        Returns:
            [type]: [description]
        """
        neg_filterd_exampled = defaultdict(list)
        n_filterd_samples = 0
        negative_count = 0
        hypo_count = 0
        # i borrowed this code from https://github.com/patverga/bran/blob/master/src/processing/utils/filter_hypernyms.py
        for doc_id in neg_doc_examples.keys():
            # get nodes for all the positive diseases
            pos_e2_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id] for pos_node in mesh_tree[pe[1]]]
            # chemical
            pos_e1_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id] for pos_node in mesh_tree[pe[0]]]

            for ne in neg_doc_examples[doc_id]:
                neg_e1 = ne[0]
                neg_e2 = ne[1]
                example_hyponyms = 0
                for neg_node in mesh_tree[ne[1]]:
                    hyponyms = [
                        pos_node for pos_node, pe in pos_e2_examples if neg_node in pos_node and neg_e1 == pe[0]
                    ]
                    example_hyponyms += len(hyponyms)
                if example_hyponyms == 0:
                    negative_count += 1
                    neg_filterd_exampled[doc_id].append((neg_e1, neg_e2))
                else:
                    hypo_count += example_hyponyms
                    n_filterd_samples += 1

        return neg_filterd_exampled, n_filterd_samples

    def get_hypernym_offset_for_token(self, token: str):

        if wn.synsets(token.lower()):
            if wn.synsets(token.lower())[0].hypernyms():
                offset = str(wn.synsets(token)[0].hypernyms()[0].offset())
            else:
                offset = str(0)
        else:
            offset = str(0)
        return offset

    def get_synset_offset_with_same_pos_with_token(self, token: str, token_pos: str):
        res = 0
        if wn.synsets(token.lower()):
            for synonym in wn.synsets(token.lower()):
                synset_pos = self.get_spacy_pos_tag_from_wordnet(synonym.pos())
                if synset_pos == token_pos:
                    res = synonym.offset()
                    break
        return str(res)

    def split(self, word):
        return [char for char in word]

    def get_spacy_pos_tag_from_wordnet(self, treebank_tag):
        if treebank_tag.startswith("j") or treebank_tag.startswith("s"):
            return "ADJ"
        elif treebank_tag.startswith("v"):
            return "VEB"
        elif treebank_tag.startswith("n"):
            return "NOUN"
        elif treebank_tag.startswith("r"):
            return "ADV"
        else:
            return ""

    def convert_tokens_to_ids(self, list_tokens, vocab):
        token_ids = []
        for token in list_tokens:
            if token not in vocab:
                token_ids.append(vocab["<UNK>"])
            else:
                token_ids.append(vocab[token])
        return token_ids

    def create_vocabs(self, list_adjacency_dict, min_freq=1):

        list_words = []
        list_rels = []
        list_poses = []
        list_chars = []
        list_hypernyms = []
        list_synsets = []

        for adjacency_dict in list_adjacency_dict:

            for pud_id in adjacency_dict.keys():
                for token, value in adjacency_dict[pud_id].items():
                    list_words.append(token.text)
                    list_poses.append(token.tag_)
                    list_chars.extend(self.split(token.text))

                    if wn.synsets(token.text.lower()):
                        if wn.synsets(token.text.lower())[0].hypernyms():
                            offset = str(wn.synsets(token.text)[0].hypernyms()[0].offset())
                        else:
                            offset = str(0)
                    else:
                        offset = str(0)

                    list_hypernyms.append(offset)

                    if wn.synsets(token.text.lower()):
                        for synonym in wn.synsets(token.text.lower()):
                            synset_pos = self.get_spacy_pos_tag_from_wordnet(synonym.pos())
                            if synset_pos == token.pos_:
                                list_synsets.append(str(synonym.offset()))
                                break

                        # assert 1==0
                    for t, rel in value:
                        list_rels.append(rel)

        word_vocab = list(set(list_words))

        word_vocab.append("<UNK>")
        word_vocab.append("<PAD>")

        word_vocab = {value: key for key, value in enumerate(word_vocab)}

        rel_vocab = list(set(list_rels))
        rel_vocab.append("<PAD>")
        rel_vocab = {value: key for key, value in enumerate(rel_vocab)}

        pos_vocab = list(set(list_poses))
        pos_vocab.append("<PAD>")
        pos_vocab = {value: key for key, value in enumerate(pos_vocab)}

        char_vocab = list(set(list_chars))
        char_vocab.append("<PAD>")
        char_vocab = {value: key for key, value in enumerate(char_vocab)}

        hypernym_vocab = list(set(list_hypernyms))
        hypernym_vocab = {v: k for k, v in enumerate(hypernym_vocab)}

        list_synsets.append(str(0))

        synonym_vocab = list(set(list_synsets))
        synonym_vocab = {v: k for k, v in enumerate(synonym_vocab)}

        print(f"word vocab: {len(word_vocab)} unique words")
        print(f"dependency rel vocab: {len(rel_vocab)} unique relations")
        print(f"char vocab: {len(char_vocab)} unique characters")
        print(f"hypernym vocab: {len(hypernym_vocab)} unique hypernym id")
        print(f"synonym vocab: {len(synonym_vocab)} unique synonym id")

        return word_vocab, rel_vocab, pos_vocab, char_vocab, hypernym_vocab, synonym_vocab

    def create_in_out_features_for_doc(
        self, in_adjacency_dict, out_adjacency_dict, doc_tokens, rel_vocab, pos_vocab, char_vocab
    ):

        all_in_nodes_idx = []
        all_out_nodes_idx = []
        all_in_edge_label_ids = []
        all_out_edge_label_ids = []
        all_poses_ids = []
        all_char_ids = []

        max_node_in = -1
        max_node_out = -1
        max_char_length = -1

        for token in doc_tokens:
            all_poses_ids.append(pos_vocab[token.tag_])

            char_ids = [char_vocab[c] for c in self.split(token.text)]

            max_char_length = max(max_char_length, len(char_ids))

            all_char_ids.append(char_ids)
        # print(out_adjacency_dict)

        # create features for incoming nodes
        for key, in_adjacent in in_adjacency_dict.items():
            key_in_ids = []
            in_edge_label_ids = []

            for token, rel in in_adjacent:
                key_in_ids.append(doc_tokens.index(token))
                in_edge_label_ids.append(rel_vocab[rel])

            max_node_in = max(max_node_in, len(key_in_ids))

            # print(key_in_ids)
            # assert 1== 0

            assert len(key_in_ids) == len(in_edge_label_ids)

            all_in_nodes_idx.append(key_in_ids)
            all_in_edge_label_ids.append(in_edge_label_ids)

        # create features for outgoing nodes
        for key, out_adjacent in out_adjacency_dict.items():

            key_out_ids = []
            out_edge_label_ids = []

            for token, rel in out_adjacent:
                key_out_ids.append(doc_tokens.index(token))
                out_edge_label_ids.append(rel_vocab[rel])

            # print(key_out_ids)

            assert len(key_out_ids) == len(out_edge_label_ids)

            max_node_out = max(max_node_out, len(key_out_ids))

            all_out_nodes_idx.append(key_out_ids)
            all_out_edge_label_ids.append(out_edge_label_ids)

        return (
            all_in_nodes_idx,
            all_out_nodes_idx,
            all_in_edge_label_ids,
            all_out_edge_label_ids,
            all_poses_ids,
            all_char_ids,
            max_node_in,
            max_node_out,
            max_char_length,
        )

    def convert_examples_to_features(
        self,
        in_adjacency_dicts,
        out_adjacency_dicts,
        entity_mapping_dicts,
        word_vocab,
        rel_vocab,
        pos_vocab,
        char_vocab,
        hypernym_vocab,
        synonym_vocab,
    ):

        all_in_nodes_idx = {}
        all_out_nodes_idx = {}
        all_in_edge_label_ids = {}
        all_out_edge_label_ids = {}
        all_doc_token_ids = {}
        all_doc_pos_ids = {}
        all_doc_char_ids = {}
        all_doc_hypernym_ids = {}
        all_doc_synonym_ids = {}

        all_enitty_mapping = {}

        max_node_in = -1
        max_node_out = -1
        max_char_length = -1

        for pud_id, in_doc_adjacency_dict in in_adjacency_dicts.items():

            doc_tokens = list(in_doc_adjacency_dict.keys())

            doc_token_texts = [tok.text for tok in doc_tokens]
            doc_token_hypernyms = [self.get_hypernym_offset_for_token(tok.text) for tok in doc_tokens]
            doc_token_synonyms = [
                self.get_synset_offset_with_same_pos_with_token(tok.text, tok.pos_) for tok in doc_tokens
            ]

            out_doc_adjacency_dict = out_adjacency_dicts[pud_id]

            (
                doc_in_nodes_idx,
                doc_out_nodes_idx,
                doc_in_edge_label_ids,
                doc_out_edge_label_ids,
                doc_poses_ids,
                doc_char_ids,
                max_doc_node_in,
                max_doc_node_out,
                max_doc_char_length,
            ) = self.create_in_out_features_for_doc(
                in_doc_adjacency_dict, out_doc_adjacency_dict, doc_tokens, rel_vocab, pos_vocab, char_vocab
            )

            doc_token_ids = self.convert_tokens_to_ids(doc_token_texts, word_vocab)
            doc_token_hypernyms_ids = self.convert_tokens_to_ids(doc_token_hypernyms, hypernym_vocab)

            # print(doc_token_synonyms)

            doc_token_synonyms_ids = self.convert_tokens_to_ids(doc_token_synonyms, synonym_vocab)

            all_doc_token_ids[pud_id] = doc_token_ids
            all_in_nodes_idx[pud_id] = doc_in_nodes_idx
            all_out_nodes_idx[pud_id] = doc_out_nodes_idx
            all_in_edge_label_ids[pud_id] = doc_in_edge_label_ids
            all_out_edge_label_ids[pud_id] = doc_out_edge_label_ids
            all_doc_pos_ids[pud_id] = doc_poses_ids
            all_doc_char_ids[pud_id] = doc_char_ids
            all_doc_hypernym_ids[pud_id] = doc_token_hypernyms_ids
            all_doc_synonym_ids[pud_id] = doc_token_synonyms_ids

            max_node_in = max(max_node_in, max_doc_node_in)
            max_node_out = max(max_node_out, max_doc_node_out)
            max_char_length = max(max_doc_char_length, max_char_length)

        max_entity_span = -1
        max_mentions = -1

        all_ner_labels = {}

        for pud_id, in_doc_adjacency_dict in in_adjacency_dicts.items():

            doc_tokens = list(in_doc_adjacency_dict.keys())
            entitty_to_tokens = defaultdict(list)

            ner_label = []

            for token in doc_tokens:
                ner_label.append(ner_vocab["O"])

            for key, mapping_tokens in entity_mapping_dicts[pud_id].items():

                _, _, _, en_type, en_id = key
                list_idx_mention = []

                count = 0
                for token in mapping_tokens:
                    list_idx_mention.append(doc_tokens.index(token))
                    max_entity_span = max(max_entity_span, len(list_idx_mention))
                    if count == 0:
                        tag = "B_" + en_type
                        ner_label[doc_tokens.index(token)] = ner_vocab[tag]
                        count += 1
                    else:
                        tag = "I_" + en_type
                        ner_label[doc_tokens.index(token)] = ner_vocab[tag]

                assert len(list_idx_mention) != 0
                entitty_to_tokens[en_id].append(list_idx_mention)

                assert len(entitty_to_tokens[en_id]) != 0
                max_mentions = max(max_mentions, len(entitty_to_tokens[en_id]))

            all_enitty_mapping[pud_id] = entitty_to_tokens
            all_ner_labels[pud_id] = ner_label
            # assert 1==0

        print("max node in: ", max_node_in)
        print("max node out: ", max_node_out)
        print("max entity spans: ", max_entity_span)
        print("max entity mentions: ", max_mentions)
        print("max characters length: ", max_char_length)

        return (
            all_doc_token_ids,
            all_in_nodes_idx,
            all_out_nodes_idx,
            all_in_edge_label_ids,
            all_out_edge_label_ids,
            all_doc_pos_ids,
            all_doc_char_ids,
            all_doc_hypernym_ids,
            all_doc_synonym_ids,
            all_enitty_mapping,
            all_ner_labels,
        )
