import argparse
import h5py
import json
import time
import tqdm
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
sys.path.append(os.curdir+'/src')
import tokenization
import model, sample, load_dataset

parser = argparse.ArgumentParser(description='Fine-tune GPT-2 on your custom dataset.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True,
                    help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='345K',
                    help='Pretrained model name')
parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1,
                    help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.0001,
                    help='Learning rate for Adam')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true',
                    help='Restrict training to the transformer blocks.')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='Optimizer. <adam|sgd>.')
parser.add_argument('--noise', type=float, default=0.0,
                    help='Add noise to input training data to regularize against typos.')
parser.add_argument('--top_k', type=int, default=40,
                    help='K for top-k sampling.')
parser.add_argument('--top_p', type=float, default=0.0,
                    help='P for top-p sampling. Overrides top_k if set > 0.')
parser.add_argument('--restore_from', type=str, default='pretrained',
                    help='Either "pretrained", "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1',
                    help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=100,
                    help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023,
                    help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=1,
                    help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=1000,
                    help='Write a checkpoint every N steps')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

vocab_path = os.curdir+'/models/345K/vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context


def main():
    args = parser.parse_args()
    hparams = model.default_hparams()

    with open(os.path.join('models', args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if args.model_name == '345M':
        args.memory_saving_gradients = True
        if args.optimizer == 'adam':
            args.only_train_transformer_layers = True

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = .45
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [args.batch_size, None])
        context_in = randomize(context, hparams, args.noise)
        output = model.model(hparams=hparams, X=context_in)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=args.sample_length,
            context=context,
            batch_size=args.batch_size,
            temperature=1.0,
            top_k=args.top_k,
            top_p=args.top_p)

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars

        if args.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        elif args.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        else:
            exit('optimizer:', args.optimizer)

        opt_grads = list(zip(tf.gradients(loss, train_vars), train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.summary.scalar('loss', loss)

        summary_lr = tf.summary.scalar('learning_rate', args.learning_rate)
        summaries = tf.summary.merge([summary_lr, summary_loss])

        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(
            var_list=all_vars,
            max_to_keep=10)

        sess.run(tf.global_variables_initializer())
        if args.restore_from == 'pretrained':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, 'pretrained'))
        elif args.restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, args.run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join('models', args.model_name))
        elif args.restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', args.model_name))
        else:
            ckpt = tf.train.latest_checkpoint(args.restore_from)
        print('Loading checkpoint', ckpt)

        if ckpt == None:
            print('ckpk is None..')

        else:
            saver.restore(sess, ckpt)

        print('Loading dataset...')

        f_path = args.dataset
        fi = h5py.File(f_path, 'r')
        data_sampler = load_dataset.Sampler_hdf5_news_cut(fi['category'])

        print('Training...')

        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, args.run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

        def generate_samples():
            print('Generating samples...')
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < args.sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: args.batch_size * [context_tokens]})
                for i in range(min(args.sample_num - index, args.batch_size)):
                    text = tokenizer.convert_ids_to_tokens(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(
                        index + 1, text)
                    all_text.append(text)
                    index += 1
            print(text)
            maketree(os.path.join(SAMPLE_DIR, args.run_name))
            with open(
                    os.path.join(SAMPLE_DIR, args.run_name,
                                 'samples-{}').format(counter), 'w') as fp:
                fp.write('\n'.join(all_text))

        def sample_batch():
            return [data_sampler.sample(args.sample_length) for _ in range(args.batch_size)]

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while True:
                if counter % args.save_every == 0:
                    save()
                if counter % args.sample_every == 0:
                    generate_samples()
                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summaries),
                    feed_dict={context: sample_batch()})

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.99 + v_loss,
                            avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.5f} avg={avg:2.5f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter += 1
        except KeyboardInterrupt:
            print('interrupted')
            save()


if __name__ == '__main__':
    main()
