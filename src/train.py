import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random

from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter


from config.cdr_config import CDRConfig
from dataset.cdr_dataset import CDRDataset
from model.cdr_model import GraphEncoder, GraphStateLSTM
from utils.metrics import compute_rel_f1, compute_NER_f1_macro, decode_ner
from utils.utils import get_mean, seed_all


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

def get_mean(lis):
    return sum(lis) / len(lis)


if __name__ == "__main__":

    seed = random.randint(0, 1e6)
    seed_all(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str, help="path to the config.json file", type=str)

    args = parser.parse_args()

    config = CDRConfig.from_json_file(args.config)
    corpus = CDRCorpus(config)
    # corpus.prepare_all_vocabs()
    (
        (
            train_all_doc_token_ids,
            train_all_in_nodes_idx,
            train_all_out_nodes_idx,
            train_all_in_edge_label_ids,
            train_all_out_edge_label_ids,
            train_all_doc_pos_ids,
            train_all_doc_char_ids,
            train_all_doc_hypernym_ids,
            train_all_doc_synonym_ids,
            train_all_entity_mapping,
            train_all_ner_labels,
        ),
        train_elmo_tensor_dict,
        trainn_flair_tensor_dict,
        train_labels,
    ) = corpus.prepare_features_for_one_dataset(
        config.train_file_path, config.train_elmo_path, config.train_flair_path
    )

    (
        (
            dev_all_doc_token_ids,
            dev_all_in_nodes_idx,
            dev_all_out_nodes_idx,
            dev_all_in_edge_label_ids,
            dev_all_out_edge_label_ids,
            dev_all_doc_pos_ids,
            dev_all_doc_char_ids,
            dev_all_doc_hypernym_ids,
            dev_all_doc_synonym_ids,
            dev_all_entity_mapping,
            dev_all_ner_labels,
        ),
        dev_elmo_tensor_dict,
        dev_flair_tensor_dict,
        dev_labels,
    ) = corpus.prepare_features_for_one_dataset(config.dev_file_path, config.dev_elmo_path, config.dev_flair_path)

    (
        (
            test_all_doc_token_ids,
            test_all_in_nodes_idx,
            test_all_out_nodes_idx,
            test_all_in_edge_label_ids,
            test_all_out_edge_label_ids,
            test_all_doc_pos_ids,
            test_all_doc_char_ids,
            test_all_doc_hypernym_ids,
            test_all_doc_synonym_ids,
            test_all_entity_mapping,
            test_all_ner_labels,
        ),
        test_elmo_tensor_dict,
        test_flair_tensor_dict,
        test_labels,
    ) = corpus.prepare_features_for_one_dataset(config.test_file_path, config.test_elmo_path, config.test_flair_path)

    dev_dataset = None
    if config.use_full:

        train_all_doc_token_ids = dict(train_all_doc_token_ids, **dev_all_doc_token_ids)
        train_all_in_nodes_idx = dict(train_all_in_nodes_idx, **dev_all_in_nodes_idx)
        train_all_out_nodes_idx = dict(train_all_out_nodes_idx, **dev_all_out_nodes_idx)
        train_all_in_edge_label_ids = dict(train_all_in_edge_label_ids, **dev_all_in_edge_label_ids)
        train_all_out_edge_label_ids = dict(train_all_out_edge_label_ids, **dev_all_out_edge_label_ids)
        train_all_pos_ids = dict(train_all_pos_ids, **dev_all_pos_ids)
        train_all_char_ids = dict(train_all_char_ids, **dev_all_char_ids)
        train_all_hypernym_ids = dict(train_all_hypernym_ids, **dev_all_hypernym_ids)
        train_all_synonym_ids = dict(train_all_synonym_ids, **dev_all_synonym_ids)
        train_elmo_dict = dict(train_elmo_dict, **dev_elmo_dict)
        train_flair_dict = dict(train_flair_dict, **dev_flair_dict)
        train_all_entity_mapping = dict(train_all_entity_mapping, **dev_all_entity_mapping)
        train_all_ner_labels = dict(train_all_ner_labels, **dev_all_ner_labels)

        train_dataset = CDRDataset(
            train_all_doc_token_ids,
            train_all_in_nodes_idx,
            train_all_out_nodes_idx,
            train_all_in_edge_label_ids,
            train_all_out_edge_label_ids,
            train_all_pos_ids,
            train_all_char_ids,
            train_all_hypernym_ids,
            train_all_synonym_ids,
            train_elmo_dict,
            train_flair_dict,
            train_all_entity_mapping,
            train_all_ner_labels,
            train_labels + dev_labels,
        )

        train_dataset.set_vocabs(
            corpus.word_vocab,
            corpus.rel_vocab,
            corpus.pos_vocab,
            corpus.hypernym_vocab,
            corpus.synonym_vocab,
            corpus.char_vocab,
        )

    else:
        train_dataset = CDRDataset(
            train_all_doc_token_ids,
            train_all_in_nodes_idx,
            train_all_out_nodes_idx,
            train_all_in_edge_label_ids,
            train_all_out_edge_label_ids,
            train_all_pos_ids,
            train_all_char_ids,
            train_all_hypernym_ids,
            train_all_synonym_ids,
            train_elmo_dict,
            train_flair_dict,
            train_all_entity_mapping,
            train_all_ner_labels,
            train_labels,
        )

        dev_dataset = CDRDataset(
            dev_all_doc_token_ids,
            dev_all_in_nodes_idx,
            dev_all_out_nodes_idx,
            dev_all_in_edge_label_ids,
            dev_all_out_edge_label_ids,
            dev_all_pos_ids,
            dev_all_char_ids,
            dev_all_hypernym_ids,
            dev_all_synonym_ids,
            dev_elmo_dict,
            dev_flair_dict,
            dev_all_entity_mapping,
            dev_all_ner_labels,
            dev_labels,
        )

    test_dataset = CDRDataset(
        test_all_doc_token_ids,
        test_all_in_nodes_idx,
        test_all_out_nodes_idx,
        test_all_in_edge_label_ids,
        test_all_out_edge_label_ids,
        test_all_pos_ids,
        test_all_char_ids,
        test_all_hypernym_ids,
        test_all_synonym_ids,
        test_elmo_dict,
        test_flair_dict,
        test_all_entity_mapping,
        test_all_ner_labels,
        test_labels,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn
    )

    if dev_dataset is not None:
        dev_loader = DataLoader(
            dev_loader, batch_size=config.batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn
        )

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
        drop_out=config.drop_out,
        encoder_hidden_size=config.encoder_hidden_size,
    )

    word2vec = corpus.load_numpy(config.word2vec_path)
    encoder.word_embedding.from_pretrained(torch.FloatTensor(word2vec), freeze=True)

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
        drop_out=config.drop_out,
    )

    weighted = torch.Tensor([1, 3.65]).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.001)

    re_criterion = nn.CrossEntropyLoss(weight=weighted)
    if config.use_ner:
        ner_criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)

    train_global_step = 0
    val_global_step = 0

    writer = SummaryWriter()

    best_f1 = -1

    for i in range(5):

        loss_epoch = []
        val_loss_epoch = []

        train_rel_loss = []
        train_ner_loss = []

        model.train()

        for train_batch in tqdm_notebook(train_loader):

            train_global_step += 1
            model.zero_grad()
            batch = [t.cuda() for t in train_batch]

            inputs = batch[:-2]
            ner_label_ids = batch[-2]
            label_ids = batch[-1]

            if config.use_ner:

                re_loss = re_criterion(re_logits, label_ids)
                ner_loss = ner_criterion(ner_logits.permute(0, 2, 1), ner_label_ids)
                total_loss = re_loss + ner_loss

                total_loss.backward()

                if train_global_step % config.gradident_accumalation_step == 0:
                    nn.utils.clip_grad_norm(model.parameters(), config.gradient_clipping)
                    optimizer.step()
                    optimizer.zero_grad()

                writer.add_scalar("Loss/train_rel_loss", re_loss.item(), val_global_step)
                writer.add_scalar("Loss/train_ner_loss", ner_loss.item(), val_global_step)

                train_rel_loss.append(rel_loss.item())
                train_ner_loss.append(ner_loss.item())

            else:
                re_logits = model(inputs)
                re_loss = re_criterion(re_logits, label_ids)
                re_loss.backward()

                if train_global_step % config.gradident_accumalation_step == 0:
                    nn.utils.clip_grad_norm(model.parameters(), config.gradient_clipping)
                    optimizer.step()
                    optimizer.zero_grad()

                writer.add_scalar("Loss/train_rel_loss", re_loss.item(), train_global_step)
                train_rel_loss.append(re_loss.item())

        scheduler.step()
        avg_train_rel_loss = get_mean(train_rel_loss)

        if len(train_ner_loss) > 0:
            avg_train_ner_loss = get_mean(train_ner_loss)
            print(f"epoch:{i+1}, train_rel_loss:{avg_train_rel_loss}, train_ner_loss:{avg_train_ner_loss}")

        else:
            print(f"epoch:{i+1}, train_rel_loss: {avg_train_rel_loss}")

        if dev_dataset is not None:

            print("Evaluate on dev set .......")
            model.eval()
            dev_rel_loss = []
            dev_ner_loss = []
            pred_list = []
            target_list = []
            ner_target_list = []
            ner_pred_list = []

            with torch.no_grad():
                for val_batch in tqdm_notebook(dev_loader):

                    val_global_step += 1

                    batch = [t.cuda() for t in val_batch]

                    inputs = batch[:-2]
                    ner_label_ids = batch[-2]
                    label_ids = batch[-1]

                    if config.use_ner:
                        ner_logits, re_logits = model(inputs)

                        re_loss = re_criterion(re_logits, label_ids)
                        ner_loss = ner_criterion(ner_logits.permute(0, 2, 1), ner_label_ids)

                        total_loss = re_loss + ner_loss
                        # for rel
                        pred_classes = torch.argmax(re_logits, dim=-1).cpu().data.numpy().tolist()
                        target_classes = label_ids.cpu().data.numpy().tolist()
                        pred_list.extend(pred_classes)
                        target_list.extend(target_classes)

                        # for ner
                        ner_pred_classes = torch.argmax(ner_logits, dim=-1).cpu().data.numpy().tolist()
                        ner_target_classes = ner_label_ids.cpu().data.numpy().tolist()

                        ner_pred_classes = decode_ner(ner_pred_classes)
                        ner_target_classes = decode_ner(ner_target_classes)

                        ner_target_list.extend(ner_target_classes)
                        ner_pred_list.extend(ner_pred_classes)

                        val_loss_epoch.append(total_loss.item())

                        writer.add_scalar("Loss/dev_rel_loss", re_loss.item(), val_global_step)
                        writer.add_scalar("Loss/dev_ner_loss", ner_loss.item(), val_global_step)

                        dev_rel_loss.append(re_loss.item())
                        dev_ner_loss.append(ner_loss.item())
                    else:
                        re_logits = model(inputs)

                        re_loss = re_criterion(re_logits, label_ids)
                        # for rel
                        pred_classes = torch.argmax(re_logits, dim=-1).cpu().data.numpy().tolist()
                        target_classes = label_ids.cpu().data.numpy().tolist()
                        pred_list.extend(pred_classes)
                        target_list.extend(target_classes)

                        val_loss_epoch.append(re_loss.item())
                        writer.add_scalar("Loss/dev_rel_loss", re_loss.item(), val_global_step)
                        dev_rel_loss.append(re_loss.item())

            # avg_train_rel_loss = get_mean(train_rel_loss)
            avg_dev_rel_loss = get_mean(dev_rel_loss)
        if len(dev_ner_loss) > 0:
            avg_dev_ner_loss = get_mean((avg_dev_ner_loss))
            print(f"epoch:{i+1}, dev_rel_loss:{avg_dev_rel_loss}, dev_ner_loss:{avg_dev_ner_loss}")
            ner_f1 = compute_NER_f1_macro(ner_pred_list, ner_target_list)
            print(f"ner f1 score:{ner_f1}")
        else:
            print(f"epoch:{i+1}, dev_rel_loss: {avg_dev_rel_loss}")

        f1 = compute_rel_f1(target_list, pred_list)
        print(f"relation f1 score: {f1}")
        if f1 > best_f1:
            best_f1 = f1
            print("performance improved .... Save best model ...")
            torch.save(model.state_dict(), f"best_model_{best_f1}.pth")


print("Evaluate on test set .......")

model.eval()
test_rel_loss = []
test_ner_loss = []
pred_list = []
target_list = []
ner_target_list = []
ner_pred_list = []

with torch.no_grad():

    for val_batch in tqdm_notebook(test_loader):

        batch = [t.cuda() for t in val_batch]
        inputs = batch[:-2]
        ner_label_ids = batch[-2]
        label_ids = batch[-1]

        if config.use_ner:
            ner_logits, re_logits = model(inputs)
            re_loss = re_criterion(re_logits, label_ids)
            ner_loss = ner_criterion(ner_logits.permute(0, 2, 1), ner_label_ids)
            total_loss = re_loss + ner_loss
            # for rel
            pred_classes = torch.argmax(re_logits, dim=-1).cpu().data.numpy().tolist()
            target_classes = label_ids.cpu().data.numpy().tolist()

            pred_list.extend(pred_classes)
            target_list.extend(target_classes)

            # for ner
            ner_pred_classes = torch.argmax(ner_logits, dim=-1).cpu().data.numpy().tolist()
            ner_target_classes = ner_label_ids.cpu().data.numpy().tolist()

            ner_pred_classes = decode_ner(ner_pred_classes)
            ner_target_classes = decode_ner(ner_target_classes)

            ner_target_list.extend(ner_target_classes)
            ner_pred_list.extend(ner_pred_classes)

            test_rel_loss.append(re_loss.item())
            test_ner_loss.append(ner_loss.item())
        else:
            re_logits = model(inputs)

            re_loss = re_criterion(re_logits, label_ids)
            # for rel
            pred_classes = torch.argmax(re_logits, dim=-1).cpu().data.numpy().tolist()
            target_classes = label_ids.cpu().data.numpy().tolist()
            pred_list.extend(pred_classes)
            target_list.extend(target_classes)

            test_rel_loss.append(re_loss.item())

# avg_train_rel_loss = get_mean(train_rel_loss)
avg_test_rel_loss = get_mean(test_rel_loss)

if len(test_ner_loss) > 0:
    avg_test_ner_loss = get_mean(test_ner_loss)
    print(f"test_rel_loss:{avg_test_rel_loss}, test_ner_loss:{avg_test_ner_loss}")

    ner_f1 = compute_NER_f1_macro(ner_pred_list, ner_target_list)
    print(f"test ner f1 score:{ner_f1}")

else:
    print(f"test_rel_loss: {avg_dev_rel_loss}")

f1 = compute_rel_f1(pred_classes, target_classes)
print("f1 score on test set: {f1} ")
