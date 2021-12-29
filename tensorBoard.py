python import numpy as np

x = np.matrix([0.01, 0.02, 0.03, 0.04, 0.05]) # 入力。1行5列。値は適当。 
w = np.matrix([[0.06, 0.07, 0.08] # 重みづけ。5行3列。値は適当。 
[0.09, 0.10, 0.11] [0.12, 0.13, 0.14] [0.15, 0.16, 0.17] [0.18, 0.19, 0.20]]) 
b = np.matrix([0.21, 0.22, 0.23]) # バイアス。1行3列。値は適当。


# ネットワーク定義
# ニューラル・ネットワークの構築はinferenceでやれとチュートリアルに書いてあった。
def inference(l0):  # l0は、ニューラル・ネットワークへの入力となる行列。今回は、28ドット×28ドットの画像データを1行784列の行列にしたもの。
    # name_scope()しておくと、後述するTensorBoardでの表示がわかりやすくになります。なお、入力でも出力でもない層を、隠れ層と呼びます。
    with tf.name_scope('hidden'):  # 「隠れ」層なので名前はhidden。
        # Variable型の、重み付けの行列を作成します。truncated_normalにすると、いい感じに分布する乱数で初期化されるらしい。
        # 引数が、行列の次元です。入力が784列なので、最初の値は784に決まります。次の値を100にして、隠れ層のニューロンの数を100にしました。
        # なお、戻り値のw1は計算式そのものなので、printしても値はわかりません（Chainerだと、printできてデバッグが楽らしい）。
        w1 = tf.Variable(tf.truncated_normal([784, 100]))

        # Variable型の、バイアスの行列を作成します。constantは定数での初期化。
        # TensorFlowのチュートリアルでは0.1に初期化したのが多かったので、何も考えずに0.1で初期化しました。ただの初期値だしね。
        # shapeが行列の次元。ニューロンの数を100にしたので、100になります。
        b1 = tf.Variable(tf.constant(0.1, shape=[100]))

        # 入力に重みを掛け算して、バイアスを足して、活性化関数に通します。
        # 活性化関数は、ReLUを使用しました。ReLUの数式の意味はわかりませんでしたけど、APIを呼び出すだけなので簡単でした。
        l1 = tf.nn.relu(tf.add(tf.matmul(l0, w1), b1))

    # 出力層。0〜9のそれぞれに反応するニューロンが欲しいので、ニューロンの数は10個にします。
    with tf.name_scope('output'):
        w2 = tf.Variable(tf.truncated_normal([100, 10]))
        b2 = tf.Variable(tf.constant(0.1, shape=[10]))
        l2 = tf.add(tf.matmul(l1, w2), b2)  # 出力の使い方を制限させないために、あえて活性化関数は入れていません。

    # 計算結果（を表現する計算グラフ）を返します。
    return l2


# 損失関数
# 損失関数はloss()でやれと、チュートリアルに書いてあった。
def loss(logits, labels):
    # 引数のlogitsはニューラル・ネットワークの出力、labelsは正解データ。
    # labelsと複数形になっているのは、後述するバッチ単位で損失関数を実行するため。
    # nn.sparse_softmax_cross_entropy_with_logitsは、どれか一つのニューロンを選ぶ（0と1の両方はありえない）場合用の関数。
    # reduce_meanでその平均を計算しているのは、これもやっぱり後述するバッチ単位で損失関数は実行されるため。
    # tf.castしているのは、正解データはint32だけど関数がint64じゃないと嫌だとわがままを言ったため。
    return with_scalar_summary(
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(labels, tf.int64)), name='loss'))

# 値を、後述するTensorBoard向けに出力させる関数。
# ここまでのコードは計算方法を定義しているだけで、logitsやlabelsなどの変数は計算をすると値が設定される「場所」を指しているに過ぎません。
# なので、printとかしても全く無駄です。だから、xxx_summaryしてTensorBoardで見ます。
# 損失関数は学習がうまく行っているかどうかの判断にとても役立つので、作成したwith_scalar_summaryを通して出力対象に設定しておきます。
def with_scalar_summary(op):
    tf.scalar_summary(op.name, op)
    return op


# 最適化
# 最適化方法はtrainingで示せと、チュートリアルに書いてあった。
def training(loss, learning_rate):
    # 引数のlossは、損失関数。learning_rateは学習レートと呼ばれるもの。
    # 学習レート変えるとうまく学習するようになるというか、調整して適切な値に設定しないと全然学習されなくて涙が出る。
    # minimizeは最小化するように頑張れという指示。
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# 評価関数
# 評価をする計算グラフはevaluationで作れと、チュートリアルに書いてあった。
def evaluation(logits, labels):
    # TrueやFalseだと計算できないのでint32にキャストして、reduce_sumで合計します。Trueは1、Falseは0になります。
    return with_scalar_summary(tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.int32), name='evaluation'))


# データ読み込み、学習させる
# TensorFlowのサンプルを使用して、MNISTのデータを読み込みます。
from tensorflow.examples.tutorials.mnist import input_data
data_sets = input_data.read_data_sets('MNIST_data/')

# 入力と正解の入れ物を作成します。次元の最初の数値がNoneになっているのは、バッチやテスト用データのサイズに合わせたいから。
inputs = tf.placeholder(tf.float32, [None, 784], name='inputs')
labels = tf.placeholder(tf.int32, [None], name='labels')

# これまでに作成した関数を使用して、計算グラフを作成します。
logits = inference(inputs)
loss = loss(logits, labels)
training = training(loss, 1.0)  # 学習レートは、試した範囲では1.0が一番良かったです。
evaluation = evaluation(logits, labels)

# tensorboardへの出力のための計算グラフを作成します。
summary = tf.merge_all_summaries()

# 作成した計算グラフの実行は、Sessionを通じて実施します。
with tf.Session() as session:
    # 変数を初期化する計算グラフを実行します。session.run(計算グラフ)で、実行できます。
    session.run(tf.initialize_all_variables())

    # tensorboardへの出力ライブラリ。
    summary_writer = tf.train.SummaryWriter('MNIST_train/', session.graph)

    # バッチを1万回実行します。本当は、いい感じに学習できたら終了にすべきなのですけど……。
    for i in range(10000):
        # TensorFlowのサンプルのMNIST読み込みライブラリは、next_batch(件数)で、指定された件数のデータを読み込みます。
        # 今回は、100件単位で学習することにしました。
        batch = data_sets.train.next_batch(100)

        # placeholderへのデータの設定は、ディクショナリを使用して実現されます。
        feed_dict = {inputs: batch[0], labels: batch[1]}

        # 学習して、損失関数の値を取得します。学習の結果は捨てて、損失関数の値だけを変数に代入します。
        _, loss_value = session.run([training, loss], feed_dict=feed_dict)

        # 時々は、TensorBoardや画面にデータを出力します。
        if i % 100 == 0 and i != 0:
            # TensorBoardにデータを出力しhます。
            summary_writer.add_summary(session.run(summary, feed_dict=feed_dict), i)
            summary_writer.flush()

            # 損失関数の値を画面に出力します。
            print('Step %4d: loss = %.2f' % (i, loss_value))

            # 訓練用データとは別のテスト用データで評価して、結果を画面に出力します。
            evaluation_value = session.run(evaluation,
                                           feed_dict={inputs: data_sets.test.images,
                                                      labels: data_sets.test.labels})
            print('Score = %d/%d, precision = %.04f' %
                  (evaluation_value, data_sets.test.num_examples, evaluation_value / data_sets.test.num_examples))
