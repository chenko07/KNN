#Library
import math
import operator

Pertama-tama, kita mendefinisikan dataset sampel sebagai sebuah dictionary dengan dua kunci: 'data' untuk menyimpan data instance dan 'labels' untuk menyimpan label atribut.

# Contoh dataset sampel
dataset = {
    'data': [
        [2, 4, 5, 'A'],
        [4, 2, 1, 'B'],
        [5, 6, 3, 'B'],
        [3, 7, 1, 'A'],
        [1, 3, 5, 'A'],
        [6, 8, 9, 'B'],
        [3, 5, 2, 'A'],
        [5, 2, 7, 'B']
    ],
    'labels': ['X1', 'X2', 'X3']
}
dataset

Fungsi euclidean_distance digunakan untuk menghitung jarak Euclidean antara dua titik dalam ruang n-dimensi, di mana n adalah jumlah atribut pada setiap instance.

# Fungsi untuk menghitung jarak Euclidean antara dua titik
def euclidean_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)):
        distance += (instance1[i] - instance2[i]) ** 2
    return math.sqrt(distance)

# Example usage
def euclidean_distance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)):
        distance += (instance1[i] - instance2[i]) ** 2
    return math.sqrt(distance)

Fungsi get_neighbors digunakan untuk mendapatkan K tetangga terdekat dari suatu instance uji pada dataset pelatihan.

# Fungsi untuk mendapatkan K tetangga terdekat
def get_neighbors(training_set, test_instance, k):
    distances = []
    for i in range(len(training_set)):
        dist = euclidean_distance(test_instance, training_set[i][:len(test_instance)])
        distances.append((training_set[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

Fungsi get_response digunakan untuk mendapatkan kelas yang paling sering muncul dari K tetangga terdekat.

# Fungsi untuk mendapatkan kelas dari K tetangga terdekat
def get_response(neighbors):
    class_votes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

Pada contoh penggunaan fungsi KNN di akhir kode, kita menggunakan instance uji [3, 5, 7] dan k=3 untuk mendapatkan prediksi kelas.

# Contoh penggunaan fungsi KNN
test_instance = [3, 5, 7]
k = 3
neighbors = get_neighbors(dataset['data'], test_instance, k)
prediction = get_response(neighbors)
print('Test instance:', test_instance)
print('Predicted class:', prediction)
