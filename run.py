from train import *
from test import *
from graph import draw
from timeit import default_timer as timer


def run():
    # 모델 선언
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    # 파라미터 초기화
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    train_losses, valid_losses = [], []
    train_best_loss = int(1e9)
    val_best_loss = int(1e9)

    # 학습
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)

        train_losses.append(train_loss)
        valid_losses.append(val_loss)

        if train_loss < train_best_loss:
            train_best_loss = train_loss
        if val_loss < val_best_loss:
            val_best_loss = val_loss
            torch.save(transformer.state_dict(), 'saved/model-{0}-{1}-{2}.pt'.format(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE))

        f = open('result/train_val_loss_log/train_loss-{0}-{1}-{2}.txt'.format(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE), 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/train_val_loss_log/valid_loss-{0}-{1}-{2}.txt'.format(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE), 'w')
        f.write(str(valid_losses))
        f.close()

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

    # 학습 model train set으로 bleu평가 진행
    result_bleu = test_model(transformer, TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE)
    draw(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE)
    f = open('result/bleu_min_train_val_loss/result-{0}-{1}-{2}.txt'.format(TUNING_LAYERS, TUNING_FFN_HID_DIM, TUNING_EMB_SIZE), 'w')
    f.write("BLEU: " + str(result_bleu) +" min_train_loss: " + str(train_best_loss) + " min_valid_loss: " + str(val_best_loss))
    f.close()


if __name__ == '__main__':
    for TUNING_LAYERS in [4, 5, 6]:
        for TUNING_FFN_HID_DIM in [512, 1024, 2048]:
            for TUNING_EMB_SIZE in [256, 512]:
                EMB_SIZE = TUNING_EMB_SIZE
                FFN_HID_DIM = TUNING_FFN_HID_DIM
                NUM_ENCODER_LAYERS = TUNING_LAYERS
                NUM_DECODER_LAYERS = TUNING_LAYERS
                run()


