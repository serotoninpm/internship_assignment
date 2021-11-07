from bleu import get_bleu
from run import *


def test_model(model, TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE):
    model.eval()
    model.load_state_dict(torch.load('saved/model-{0}-{1}-{2}.pt'.format(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE)))

    test_iter = Papago_Dataset('./dataset/test_parallel.csv')
    test_dataloader = DataLoader(test_iter, batch_size=1, collate_fn=collate_fn)

    bleu_set = []
    result_bleu = []

    for src, tgt in test_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
        source_sentence = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(src.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        truth_sentence = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        predict_sentence = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
        # print('source : ', source_sentence)
        # print('truth :', truth_sentence)
        # print('predicted :', predict_sentence)
        bleu = get_bleu(hypotheses=[predict_sentence.split()], reference=[truth_sentence.split()])
        bleu_set.append(bleu)
        # print("Bleu score:",bleu)
        # print()


    result_bleu.append(sum(bleu_set)/len(bleu_set))
    print("BLEU", result_bleu[0])
    return result_bleu[0]


# 탐욕(greedy) 알고리즘을 사용하여 출력 순서(sequence)를 생성하는 함수
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# 입력 문장을 도착어로 번역하는 함수
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


if __name__ == '__main__':
    torch.manual_seed(0)
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    transformer = transformer.to(DEVICE)
    test_model(transformer)

