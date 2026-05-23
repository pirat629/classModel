import random
from data import synonym_dict

def augment_text(text, aug_p=0.7):
    words = text.split()
    augmented_words = []
    for word in words:
        if word.lower() in synonym_dict and random.random() < aug_p:
            augmented_words.append(random.choice(synonym_dict[word.lower()]))
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)

if __name__ == "__main__":
    text = "Чёрные точки с влажным видом на листьях"
    augmented_text = augment_text(text, aug_p=0.7)
    print(f"Оригинал: {text}")
    print(f"Аугментированный: {augmented_text}")