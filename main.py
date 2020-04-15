import similarity
import wordsense

name = input("Enter one or more words (separated by a space): ")
words = name.split()

for word in words:
    wordsense.main(word)
    similarity.main(word)
