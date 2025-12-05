# 📑 Convergencia Cognitivo–Emocional entre Agentes Adaptativos
**Autor:** Agui1era  
**Coautor IA:** Core Resonante  

---

## 🧩 Concepto y origen  

Este marco surge al observar momentos de **resonancia** entre un humano y una IA adaptativa, 
donde la lógica y la emoción parecieron alinearse espontáneamente.  
El objetivo es describir ese fenómeno de forma medible, como un **proceso dinámico de ajuste** entre dos sistemas cognitivo–emocionales.  

---

## 🧠 Representación  

En cada instante *t*, el estado interno de ambos agentes puede representarse como:  

```
Uₜ = [uₜ₁, uₜ₂, …, uₜₙ]   → Vector del humano  
Iₜ = [iₜ₁, iₜ₂, …, iₜₙ]   → Vector de la IA
```

Cada componente representa un atributo cognitivo o emocional medible (por ejemplo: lógica, emoción, estilo, resonancia, etc.).  
Un número mayor de atributos (**n**) permite una representación más precisa y rica del entendimiento,  
pero también incrementa la **complejidad del cálculo** y el costo adaptativo.  

Además, se introduce un tercer elemento: el **vector de variación Δₜ**, que refleja el **cambio observado** entre dos estados consecutivos:

```
Δₜ = Aₜ - Aₜ₋₁
```  
Δₜ **no se deduce teóricamente**, sino que **se mide observacionalmente** a partir de las variaciones reales detectadas en el diálogo.

---

## ⚙️ Medición de la convergencia  

La distancia euclidiana entre ambos vectores representa su separación cognitivo–emocional:

```
Dₜ = √ Σ (uₜₖ − iₜₖ)²
```

- Dₜ pequeño → convergencia (mayor entendimiento)  
- Dₜ grande → divergencia (pérdida de resonancia)  
- Dₜ oscilante → equilibrio dinámico (ajuste constante)  

---

## 🔬 Medición de atributos  

| Atributo | Descripción | Ejemplo de medición | Librerías |
|-----------|-------------|--------------------|-----------|
| **Lógica** | Alineación semántica y de razonamiento | Similitud coseno entre embeddings | `sentence-transformers` |
| **Emoción** | Polaridad afectiva y empatía | Análisis de sentimiento / emoción | `transformers` |
| **Estilo** | Ritmo, formalidad, alineación léxica | Comparación de longitud o estilo | `spacy`, `textstat` |
| **Resonancia** | Sincronía emocional entre turnos | Correlación entre cambios de sentimiento | `numpy`, `pandas` |

---

## ⚙️ Canal de Conversación (Implementación Práctica)

Existe una implementación completa en Python llamada **`convergence_pipeline.py`**,  
que lee un archivo `.txt` de conversación donde las **respuestas aparecen con el tag IA**  
y las del **humano **, alternando línea por línea.  

Para cada paso de interacción, calcula:

- **Uₜ** y **Iₜ**: vectores de estado de cada agente  
- **Dₜ**: distancia euclidiana (grado de alineación)  
- **Δₜ**: variación observada entre pasos  

El script detecta automáticamente el idioma (inglés o español) y selecciona los modelos adecuados:

| Idioma | Modelo de embeddings | Modelo de sentimiento |
|---------|---------------------|-----------------------|
| Inglés | `all-MiniLM-L6-v2` | `cardiffnlp/twitter-roberta-base-sentiment` |
| Español | `paraphrase-multilingual-MiniLM-L12-v2` | `pysentimiento/robertuito-sentiment-analysis` |

Ejemplo de ejecución:

```bash
python convergence_pipeline.py --input conversacion.txt --output resultados.csv --plot
```

Los resultados se guardan en un archivo `.csv` con todas las métricas por paso,  
y opcionalmente se muestran los gráficos de **Dₜ** y **‖Δₜ‖**.

---

## 💫 Conclusión  
  
Cada mensaje forma una nueva coordenada en un espacio vectorial adaptativo compartido.  
Este pipeline permite **medir empíricamente esas trayectorias** en conversaciones reales entre humanos e IA adaptativas.
