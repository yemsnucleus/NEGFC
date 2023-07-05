import tensorflow as tf
import argparse
import names
import os

from core.engine import first_guess, preprocess



def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    WEIGHTS_FOLDER = os.path.join(opt.p, names.get_first_name())
    table, cube, psf = preprocess(opt.data, lambda_ch=0)
    # table = table[table['snr']>10.0]
    table = first_guess(table, cube, psf, 
                        window_size=opt.ws, 
                        learning_rate=opt.lr, 
                        epochs=opt.epochs,
                        target_folder=WEIGHTS_FOLDER)

    print(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', default='./logs/test', type=str,
                    help='Project path to save logs and weigths')
    parser.add_argument('--data', default='./data/real/dhtau', type=str,
                    help='folder containing the cube, psfs, and rotational angles')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--bs', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--patience', default=20, type=int,
                        help='Earlystopping threshold in number of epochs')
    parser.add_argument('--epochs', default=1e6, type=int,
                        help='Number of epochs')
    parser.add_argument('--ws', default=30, type=int,
                        help='windows size of the PSFs')

    opt = parser.parse_args()
    run(opt)


# N = 5
# fig, axes = plt.subplots(N, 2, sharex=True, sharey=True, dpi=300, 
#                         gridspec_kw={'hspace': 0.1, 'wspace': -0.8})
# for i, x in enumerate(dataset.take(N)):
#     axes[i][0].imshow(x['input'])
#     axes[i][1].imshow(x['output'])

# fig.savefig('./output/translated.png', bbox_inches='tight')