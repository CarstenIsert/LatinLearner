import embedding
import text_handling

data = text_handling.TextData()
texts = data.load_tokenized_data('small_library')
int_text, int2word, vocabulary_size = data.generate_tokenized_dataset(texts)
print(len(int_text))
model = embedding.WordEmbedding(int_text, int2word, vocabulary_size)
model.process()

