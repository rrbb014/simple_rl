import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# 밴딧의 손잡이 목록 작성
# 현재 손잡이 인덱스 3 이 가장 자주 +1 보상이 나오도록 설정되어있음.

bandit_arms = [0.2, 0, -0.2, -2]
num_arms = len(bandit_arms)

def pull_bandit(bandit):
	# Get random value
	result = np.random.randn(1)
	if result > bandit:
		return 1
	else:
		return -1

if __name__ == "__main__":
	tf.reset_default_graph()  # Clears the default graph stack and resets the global default graph.
	
	# Feed-forward 구현
	weights = tf.Variable(tf.ones([num_arms]))
	output = tf.nn.softmax(weights)
	
	# 학습 과정
	# reward, action을 네트워크에 피드 -> 비용계산
	# 비용을 이용해 업데이트
	
	reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
	action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
	
	responsible_output = tf.slice(output, action_holder, [1])
	loss = -(tf.log(responsible_output) * reward_holder)
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
	update = optimizer.minimize(loss)
	
	# 에이전트 학습할 총 에피소드 수
	total_episodes = 1000
	# 밴딧 보상을 0으로 초기화
	total_reward = np.zeros(num_arms)
	
	init = tf.global_variables_initializer()
	
	# computational graph 런칭
	with tf.Session() as sess:
		sess.run(init)
		i = 0
		while i < total_episodes:
			# 볼츠만 분포에 따라 액션 선택
			actions = sess.run(output)
			a = np.random.choice(actions, p=actions)
			action = np.argmax(actions == a)
			# 밴딧 손잡이 중 하나를 선택, 보상받기
			reward = pull_bandit(bandit_arms[action])
			
			# 네트워크 업데이트.
			_, resp, ww = sess.run([update, responsible_output, weights], \
									feed_dict ={reward_holder:[reward], action_holder: [action]})
			
			# 보상 총계 업데이트
			total_reward[action] += reward
			if i % 50 == 0:
				print("Running reward for the " + str(num_arms) + " arms of the bandit: " + str(total_reward))
			
			i += 1
			
		print('\nThe agent thinks arm ' + str(np.argmax(ww)+1) + " is the most promising...")
		if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
			print('... and it was right')
		else:
			print('... adn it was Wrong')
			
	# TODO : tensorboadrd implementation