# Neural Network Sample

Dieses Projekt definiert ein einfaches neuronales Netzwerk mit einer Schicht, trainiert es mit einem grundlegenden Datensatz und testet es mit einer neuen Eingabe. Das Netzwerk verwendet die `tanh`-Aktivierungsfunktion und deren Ableitung für das Training.

## Anforderungen

- Python 3.x
- NumPy

## Installation

Stelle sicher, dass du Python 3.x und NumPy installiert hast. Du kannst NumPy mit pip installieren:

```bash
pip install numpy
```

## Beschreibung
### Initialisierung: 
Das neuronale Netzwerk wird mit einer zufälligen Gewichtsmatrix der Größe 3x1 initialisiert. Der Zufallsgenerator wird auf 1 gesetzt, um die Reproduzierbarkeit sicherzustellen. Eine Lernrate (learning_rate) wird verwendet, um die Größe der Gewichtsanpassungen zu steuern.
### Aktivierungsfunktion: 
Die tanh-Funktion wird als Aktivierungsfunktion verwendet.
### Ableitung der Aktivierungsfunktion: 
Die Ableitung der tanh-Funktion wird für das Backpropagation-Training verwendet.
### Vorwärtspropagation: 
Berechnet die Ausgabe des Netzwerks durch Anwendung der tanh-Funktion auf das Skalarprodukt der Eingaben und der Gewichtsmatrix.
### Training: 
Das Netzwerk wird mit den bereitgestellten Trainingsdaten trainiert. Der Trainingsprozess umfasst Vorwärtspropagation, Fehlerberechnung und Anpassung der Gewichte.
### Testen: 
Das Netzwerk wird mit einem neuen Eingabebeispiel getestet.
### Lizenz
Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe die LICENSE Datei für weitere Details.