# Guía Completa de Instalación, Ejecución y Pruebas de OMPar

## Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación Paso a Paso](#instalación-paso-a-paso)
- [Conversión del Modelo](#conversión-del-modelo)
- [Ejecución](#ejecución)
- [Pruebas y Validación](#pruebas-y-validación)
- [Solución de Problemas](#solución-de-problemas)
- [Estructura del Proyecto](#estructura-del-proyecto)

---

## Descripción General

**OMPar** es una herramienta de compilación orientada a compiladores diseñada para identificar y generar oportunidades de paralelización para código serial utilizando IA. El sistema consta de dos componentes principales:

1. **OMPify**: Detecta oportunidades de paralelización en código
2. **MonoCoder**: Genera los pragmas OpenMP apropiados cuando se identifica que un bucle for se beneficiaría de la paralelización

### Paper Asociado
**Título**: OMPar: Automatic Parallelization with AI-Driven Source-to-Source Compilation

### Rendimiento
En el conjunto de pruebas HeCBench (770 bucles), OMPar logra:
- **74%** de precisión en predicción de pragmas
- **86%** de precisión con verificación de compilación y ejecución
- Supera significativamente a AutoPar (56%) e ICPC (62%)

---

## Requisitos del Sistema

### Hardware
- **GPU**: NVIDIA con soporte CUDA (recomendado para mejor rendimiento)
- **RAM**: Mínimo 16 GB (recomendado 32 GB)
- **Almacenamiento**: ~10 GB de espacio libre

### Software
- **Sistema Operativo**: Linux (probado en Ubuntu 24.04)
- **Python**: 3.11 o 3.12
- **CUDA**: 12.1 o superior (probado con CUDA 13.0)
- **Git**: Para clonar repositorios

### Verificación de Requisitos

```bash
# Verificar Python
python3 --version
# Salida esperada: Python 3.12.x o 3.11.x

# Verificar CUDA
nvcc --version
# Salida esperada: Cuda compilation tools, release 12.1 o superior

# Verificar pip
pip3 --version
```

---

## Instalación Paso a Paso

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Scientific-Computing-Lab/OMPar
cd OMPar
```

### 2. Instalar Dependencias del Sistema

```bash
# Instalar python3-venv (necesario para crear entornos virtuales)
sudo apt update
sudo apt install -y python3.12-venv
```

### 3. Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv ompar_env

# Activar entorno virtual
source ompar_env/bin/activate
```

> **Nota**: Siempre debes activar el entorno virtual antes de usar OMPar.

### 4. Actualizar pip

```bash
pip install --upgrade pip
```

### 5. Instalar PyTorch con Soporte CUDA

```bash
# Instalar PyTorch 2.5.1 con CUDA 12.1
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Importante**: PyTorch 2.5.1 es la versión más reciente compatible con CUDA 12.1. Versiones anteriores tienen vulnerabilidades de seguridad.

### 6. Instalar Dependencias de Python

```bash
# Instalar paquetes principales
pip install transformers datasets tokenizers huggingface_hub safetensors tqdm requests pyyaml regex

# Instalar tree-sitter versión específica
pip install tree-sitter==0.20.4
```

> **Crítico**: Debe ser tree-sitter 0.20.4 exactamente. Versiones más recientes tienen cambios incompatibles en la API.

### 7. Compilar el Parser

```bash
cd parser

# Eliminar versión anterior si existe
rm -rf vendor/tree-sitter-c-sharp

# Clonar tree-sitter-c-sharp versión compatible
cd vendor
git clone https://github.com/tree-sitter/tree-sitter-c-sharp
cd tree-sitter-c-sharp
git checkout v0.20.0
cd ../..

# Compilar el parser
source ../ompar_env/bin/activate
python build.py

# Verificar que se creó el archivo
ls -lh my-languages.so
# Salida esperada: -rwxrwxr-x ... 4.9M ... my-languages.so
```

### 8. Descargar Pesos del Modelo OMPify

Los pesos del modelo no están incluidos en el repositorio. Descárgalos desde:

**Google Drive**: [OMPify Weights](https://drive.google.com/drive/folders/1tnJf9YvjpDLktVi23TkW-rpjqfdZoybf?usp=sharing)

```bash
# Descomprimir en el directorio 'model'
cd /home/antony/Desktop/paper/OMPar
# Asegúrate de que la estructura sea:
# model/
# ├── data/
# │   ├── 0
# │   ├── 1
# │   └── ... (205 archivos)
# ├── data.pkl
# └── version
```

---

## Conversión del Modelo

El modelo OMPify se descarga en formato distribuido de PyTorch. Necesitas convertirlo a `model.bin` para su uso:

### Script de Conversión

El script `convert_model.py` ya está incluido en el repositorio. Ejecútalo:

```bash
source ompar_env/bin/activate
python convert_model.py
```

**Salida esperada:**
```
Loading model from model...
✓ Model loaded successfully!
  Type: <class 'collections.OrderedDict'>
  Number of parameters: 205
  Sample keys: ['encoder.roberta.embeddings.word_embeddings.weight', ...]

✓ Saved model to model/model.bin
  File size: 477.82 MB
```

### Verificación

```bash
ls -lh model/model.bin
# Salida esperada: -rw-rw-r-- ... 477M ... model.bin
```

---

## Modificaciones de Código Necesarias

### ¿Por qué son necesarias?

PyTorch 2.5.1 tiene restricciones de seguridad que requieren PyTorch 2.6+ para cargar modelos en formato pickle. Como PyTorch 2.6+ no está disponible para CUDA 12.1, usamos el formato safetensors.

### 1. Modificar `OMPify/model.py`

**Líneas 130-133**, cambiar:
```python
model = RobertaForSequenceClassification.from_pretrained(base_model, config=self.config)    
self.model=Model(model,self.config,self.tokenizer)
self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.bin')))
```

**Por:**
```python
# Use safetensors to avoid PyTorch 2.6+ requirement
model = RobertaForSequenceClassification.from_pretrained(base_model, config=self.config, use_safetensors=True)    
self.model=Model(model,self.config,self.tokenizer)
self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.bin'), weights_only=True))
```

### 2. Modificar `compAI.py`

**Línea 14**, cambiar:
```python
self.model_gen = GPTNeoXForCausalLM.from_pretrained('MonoCoder/MonoCoder_OMP').to(device)
```

**Por:**
```python
self.model_gen = GPTNeoXForCausalLM.from_pretrained('MonoCoder/MonoCoder_OMP', use_safetensors=True).to(device)
```

> **Nota**: Estas modificaciones ya están aplicadas si seguiste esta guía.

---

## Ejecución

### Activar Entorno

Siempre activa el entorno virtual antes de ejecutar OMPar:

```bash
cd /home/antony/Desktop/paper/OMPar
source ompar_env/bin/activate
```

### Ejecutar Casos de Uso de Ejemplo

```bash
python run_ompar.py --model_weights model
```

### Uso Programático

```python
import torch
from compAI import OMPAR
import argparse

# Configurar argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='tokenizer/gpt/gpt_vocab/gpt2-vocab.json')
parser.add_argument('--merge_file', default='tokenizer/gpt/gpt_vocab/gpt2-merges.txt')
parser.add_argument('--model_weights', default='model')
args = parser.parse_args([])

# Inicializar OMPar
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ompar = OMPAR(model_path=args.model_weights, device=device, args=args)

# Analizar código
code = """for(int i = 0; i < 1000; i++){
    sum += array[i];
}"""

pragma = ompar.auto_comp(code)
if pragma:
    print(f"Pragma sugerido: #pragma {pragma}")
else:
    print("No se requiere paralelización")
```

---

## Pruebas y Validación

### Casos de Prueba Incluidos

El archivo `use_cases.jsonl` contiene 7 casos de prueba:

#### 1. Bucle Paralelo Simple ✅
```c
for (int i = 0; i < size; i++) {
    array[i] = compute(array[i]);
}
```
- **Esperado**: `#pragma omp parallel for`
- **Predicho**: `#pragma omp parallel for` ✓

#### 2. Suma de Arrays ✅
```c
for (int i = 0; i < size; i++) {
    result[i] = array1[i] + array2[i];
}
```
- **Esperado**: `#pragma omp parallel for`
- **Predicho**: `#pragma omp parallel for` ✓

#### 3. Reducción ✅
```c
for (int i = 0; i < size; i++) {
    sum += array[i];
}
```
- **Esperado**: `#pragma omp parallel for reduction(+:sum)`
- **Predicho**: `#pragma omp parallel for reduction( + :sum)` ✓

#### 4. Diagonal de Matriz ⚠️
```c
for (int i = 0; i < size; i++) {
    matrix[i][i] = value;
}
```
- **Esperado**: `#pragma omp parallel for`
- **Predicho**: (vacío) - Falso negativo

#### 5. Multiplicación por Escalar ✅
```c
for (int i = 0; i < size; i++) {
    temp[i] = array[i] * factor;
}
```
- **Esperado**: `#pragma omp parallel for`
- **Predicho**: `#pragma omp parallel for` ✓

#### 6. Early Return (No Paralelizable) ✅
```c
for (int i = 0; i < size; i++) {
    if (array[i] < 0) {
        return i;
    }
}
```
- **Esperado**: (vacío)
- **Predicho**: (vacío) ✓

#### 7. Dependencia de Datos (No Paralelizable) ✅
```c
for (int i = 0; i < size; i++) {
    array[i] = array[i - 1] + array[i + 1];
}
```
- **Esperado**: (vacío)
- **Predicdo**: (vacío) ✓

### Resultados de las Pruebas

**Precisión**: 6/7 (85.7%)
- ✅ Verdaderos Positivos: 4
- ✅ Verdaderos Negativos: 2
- ⚠️ Falsos Negativos: 1
- ✅ Falsos Positivos: 0

### Ejecutar Pruebas Personalizadas

Crea un archivo `custom_tests.jsonl`:

```json
{"code": "for (int i = 0; i < n; i++) { a[i] = b[i] + c[i]; }", "label": true, "pragma": "#pragma omp parallel for"}
{"code": "for (int i = 1; i < n; i++) { a[i] = a[i-1] + 1; }", "label": false, "pragma": ""}
```

Modifica `run_ompar.py` para usar tu archivo:

```python
with open('custom_tests.jsonl', 'r') as f:
    # ... resto del código
```

---

## Solución de Problemas

### Error: `conda: command not found`

**Solución**: Usar `venv` en lugar de conda (ya implementado en esta guía).

### Error: `tree-sitter` versión incompatible

```
ValueError: Incompatible Language version 15. Must be between 13 and 14
```

**Solución**:
```bash
cd parser/vendor
rm -rf tree-sitter-c-sharp
git clone https://github.com/tree-sitter/tree-sitter-c-sharp
cd tree-sitter-c-sharp
git checkout v0.20.0
cd ../..
python build.py
```

### Error: PyTorch 2.6+ requerido

```
ValueError: Due to a serious vulnerability issue in `torch.load`...
```

**Solución**: Aplicar las modificaciones de código descritas en la sección [Modificaciones de Código Necesarias](#modificaciones-de-código-necesarias).

### Error: CUDA out of memory

**Solución**: Reducir el tamaño del batch o usar CPU:

```python
device = 'cpu'  # Forzar uso de CPU
```

### Warnings sobre `attention_mask`

```
The attention mask and the pad token id were not set...
```

**Solución**: Estos warnings son normales y no afectan la funcionalidad. Puedes ignorarlos.

---

## Estructura del Proyecto

```
OMPar/
├── ompar_env/                    # Entorno virtual Python
├── parser/
│   ├── my-languages.so          # Parser compilado (4.9 MB)
│   ├── vendor/
│   │   └── tree-sitter-c-sharp/ # v0.20.0
│   ├── build.py
│   └── build.sh
├── OMPify/
│   ├── model.py                 # Modelo de clasificación
│   └── __init__.py
├── model/
│   ├── data/                    # Pesos distribuidos (205 archivos)
│   ├── data.pkl                 # Índice del modelo
│   ├── model.bin                # Modelo convertido (477.82 MB)
│   └── version
├── tokenizer/
│   └── gpt/
│       └── gpt_vocab/
│           ├── gpt2-vocab.json
│           └── gpt2-merges.txt
├── evaluation/                  # Scripts de evaluación
├── compAI.py                    # Clase principal OMPAR
├── run_ompar.py                 # Script de ejecución
├── convert_model.py             # Script de conversión de modelo
├── use_cases.jsonl              # Casos de prueba
├── requirements.txt             # Dependencias simplificadas
├── environment.yml              # Entorno conda original
├── README.md                    # README original
└── INSTALLATION_GUIDE.md        # Esta guía
```

---

## Información Adicional

### Modelos Utilizados

1. **OMPify**: 
   - Base: `microsoft/graphcodebert-base`
   - Tamaño: 477.82 MB
   - Propósito: Clasificación de paralelización

2. **MonoCoder**:
   - Base: `MonoCoder/MonoCoder_OMP`
   - Tamaño: 3.59 GB
   - Propósito: Generación de pragmas

### Recursos del Sistema Durante la Ejecución

- **Memoria GPU**: ~8-10 GB (con CUDA)
- **Memoria RAM**: ~6-8 GB
- **Tiempo de carga inicial**: ~30-60 segundos
- **Tiempo por predicción**: ~2-5 segundos

### Licencia

MIT License - Ver archivo LICENSE en el repositorio

### Citas

Si usas OMPar en tu investigación, por favor cita:

```bibtex
@article{ompar2024,
  title={OMPar: Automatic Parallelization with AI-Driven Source-to-Source Compilation},
  author={...},
  journal={...},
  year={2024}
}
```

### Contacto y Soporte

- **Repositorio**: https://github.com/Scientific-Computing-Lab/OMPar
- **Issues**: https://github.com/Scientific-Computing-Lab/OMPar/issues

---

## Resumen de Comandos Rápidos

```bash
# Instalación completa
git clone https://github.com/Scientific-Computing-Lab/OMPar
cd OMPar
sudo apt install -y python3.12-venv
python3 -m venv ompar_env
source ompar_env/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tokenizers huggingface_hub safetensors tqdm requests pyyaml regex tree-sitter==0.20.4

# Compilar parser
cd parser
rm -rf vendor/tree-sitter-c-sharp
cd vendor && git clone https://github.com/tree-sitter/tree-sitter-c-sharp && cd tree-sitter-c-sharp && git checkout v0.20.0 && cd ../..
python build.py
cd ..

# Convertir modelo (después de descargar pesos)
python convert_model.py

# Ejecutar
python run_ompar.py --model_weights model
```

---

**Última actualización**: 4 de enero de 2026  
**Versión de la guía**: 1.0  
**Sistema probado**: Ubuntu 24.04, Python 3.12.3, CUDA 13.0
