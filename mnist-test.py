# TensorFlow をプログラムにインポート
import tensorflow as tf

# MNIST データセットをロードして準備
# サンプルを整数から浮動小数点数に変換
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 層を積み重ねてtf.keras.Sequentialモデルを構築
# 訓練のためにオプティマイザと損失関数を選ぶ
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# 各サンプルについて、モデルは "logits" または "log-odds" スコアのベクトルをクラスごとに返す
predictions = model(x_train[:1]).numpy()
predictions

# tf.nn.softmax 関数は、クラスごとにこれらのロジットを "probabilities" に変換
tf.nn.softmax(predictions).numpy()

# losses.SparseCategoricalCrossentropy 損失は、ロジットのベクトルと True インデックスを取り、各サンプルのスカラー損失を返す
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# この損失は、True のクラスの負の対数確率に等しくなる
# モデルが正しいクラスであることが確実な場合はゼロになる。
# トレーニングされていないこのモデルでは、ランダムに近い確率（クラス当たり 1/10）が得られるため、最初の損失は -tf.math.log(1/10) ~= 2.3 に近くなる。
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# 損失を最小限に抑えられるようにモデルのパラメータを Model.fit メソッドで調整
model.fit(x_train, y_train, epochs=5)

# Model.evaluate メソッドは通常、検証セットがテストセットでモデルのパフォーマンスをチェックする
model.evaluate(x_test,  y_test, verbose=2)

# tf.nn.softmax 関数はクラスごとにこれらのロジットを "確率" に変換する
# モデルが確率を返すようにするには、トレーニング済みのモデルをラップして、それに softmax を接続することができる
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])


# array([[ 0.06119704, -0.20086515,  0.00702903, -0.48996255, -0.45208266,
#         -0.5741232 ,  0.9203826 , -0.15006919, -0.01316312, -0.61534005]],
#       dtype=float32)

# tf.nn.softmax(predictions).numpy()

