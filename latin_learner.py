import embedding

data = embedding.TextData()
texts = data.read_data('small_library')
int_text, int2word, vocabulary_size = data.generate_dataset(texts)
print(len(int_text))
model = embedding.WordEmbedding(int_text, int2word, vocabulary_size)
model.process()

