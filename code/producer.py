from kafka import KafkaProducer, KafkaConsumer
import json

# 1. 创建 Kafka 生产者
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])


# 2. 创建 Kafka 消费者
consumer = KafkaConsumer('my_topic', bootstrap_servers=['localhost:9092'])

# 3. 读取 JSON 文件
with open(r'C:\Users\Lenovo\Desktop\大数据软件\project2\dev-v2.0 (1).json') as f:
    data = json.load(f)

# 4. 将数据写入 Kafka 主题
for item in data:
    producer.send('my_topic', json.dumps(item).encode('utf-8'))
    time.sleep(0.5)

# 5. 从 Kafka 主题中读取数据并处理
for message in consumer:
    print(message.value.decode('utf-8'))
    
    # 在这里添加处理逻辑
