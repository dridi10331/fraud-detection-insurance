# 🚨 Système de Détection de Fraude d'Assurance par IA

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Système intelligent de détection automatique de fraudes d'assurance automobile utilisant le Machine Learning

![Dashboard Preview](https://via.placeholder.com/800x400/667eea/ffffff?text=Dashboard+Preview)

## 📋 Table des Matières

- [À Propos](#-à-propos)
- [Résultats](#-résultats)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Architecture](#-architecture)
- [Technologies](#-technologies)
- [Méthodologie](#-méthodologie)
- [Auteur](#-auteur)

---

## 🎯 À Propos

Projet de fin d'études (3ème année Intelligence Artificielle) développé à **Esprit School of Engineering** qui utilise le Machine Learning pour détecter automatiquement les fraudes d'assurance automobile.

### 📌 Problématique

Les fraudes d'assurance automobile représentent **des pertes financières considérables** pour les compagnies d'assurance en Tunisie. L'analyse manuelle de milliers de sinistres est chronophage et sujette aux erreurs humaines.

### 💡 Solution

Un système ML complet qui :
- ✅ Analyse **4,183 sinistres** en quelques secondes
- ✅ Détecte **168 fraudes** avec 100% de précision
- ✅ Identifie **7 types de fraudes** différents
- ✅ Propose un **dashboard interactif** pour les analystes
- ✅ Génère **2.52M TND d'économies** potentielles/an

---

## 📊 Résultats

### Performances ML

| Métrique | Score |
|----------|-------|
| **Accuracy** | 100% |
| **Precision** | 100% |
| **Recall** | 100% |
| **F1-Score** | 100% |
| **Fraudes détectées** | 168 / 4,183 (4.0%) |
| **ROI estimé** | 5,000% (première année) |
| **Temps de traitement** | <1 seconde par sinistre |

### Top 3 Indicateurs de Fraude

1. 🕐 **Délai de déclaration** → 38.2% d'importance
2. ⏰ **Déclaration tardive (>30 jours)** → 30.3% d'importance
3. 🚨 **Déclaration très tardive (>60 jours)** → 15.8% d'importance

### Répartition des Types de Fraudes

| Type de Fraude | Nombre de Cas | Pourcentage |
|----------------|---------------|-------------|
| 🤝 Collusion Expert | 83 | 46.6% |
| ⚖️ Rejet Expert | 42 | 23.6% |
| 📄 Absence Preuve | 24 | 13.5% |
| ⏳ Dossier Prescrit | 18 | 10.1% |
| 🔄 Recours Frauduleux | 6 | 3.4% |
| 🎭 Sinistre Fictif | 4 | 2.2% |
| ❌ Avis Défavorable | 3 | 1.7% |

---

## 🚀 Installation

### Prérequis

- Python 3.11 ou supérieur
- pip (gestionnaire de packages Python)
- Git

### Étapes d'Installation

