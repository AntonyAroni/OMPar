# Guía para Subir OMPar a tu Propio Repositorio

## 📝 Pasos para Crear tu Repositorio

### 1. Crear Repositorio en GitHub/GitLab

**En GitHub:**
1. Ve a https://github.com/new
2. Nombre del repositorio: `OMPar` (o el que prefieras)
3. Descripción: "OMPar - Automatic Parallelization with AI-Driven Source-to-Source Compilation"
4. **NO** inicialices con README, .gitignore o licencia (ya los tienes)
5. Click en "Create repository"

**En GitLab:**
1. Ve a https://gitlab.com/projects/new
2. Sigue pasos similares

### 2. Configurar Git Local

```bash
cd /home/antony/Desktop/paper/OMPar

# Verificar estado actual
git status

# Si no está inicializado, inicializar
git init

# Configurar tu información (si no lo has hecho)
git config user.name "Tu Nombre"
git config user.email "tu-email@ejemplo.com"
```

### 3. Cambiar el Remote al Tuyo

```bash
# Ver remote actual
git remote -v

# Eliminar remote original (si existe)
git remote remove origin

# Agregar tu remote (reemplaza con tu URL)
git remote add origin https://github.com/TU-USUARIO/OMPar.git
# O para GitLab:
# git remote add origin https://gitlab.com/TU-USUARIO/OMPar.git

# Verificar
git remote -v
```

### 4. Preparar Archivos para Commit

```bash
# Ver qué archivos se incluirán
git status

# Agregar todos los archivos (respetando .gitignore)
git add .

# Ver qué se va a commitear
git status
```

### 5. Crear Commit

```bash
# Commit con mensaje descriptivo
git commit -m "Initial commit: OMPar setup with installation guides and cleanup scripts"

# O commit más detallado
git commit -m "Initial commit: OMPar AI-driven parallelization tool

- Added comprehensive installation guide (INSTALLATION_GUIDE.md)
- Added cleanup guide for disk space management
- Updated .gitignore to exclude virtual env and large model files
- Modified code to use safetensors for PyTorch 2.5.1 compatibility
- Compiled parser with tree-sitter-c-sharp v0.20.0
- Tested successfully with 7 use cases (85.7% accuracy)"
```

### 6. Subir a tu Repositorio

```bash
# Primera vez (crear rama main y subir)
git branch -M main
git push -u origin main

# Siguientes veces
git push
```

---

## 📦 Qué se Incluirá en el Repositorio

### ✅ Archivos Incluidos (~100-200 MB):
- Código fuente (`.py` files)
- Guías de instalación y limpieza
- Tokenizadores
- Archivos de configuración
- Scripts de ejemplo

### ❌ Archivos Excluidos (por .gitignore):
- `ompar_env/` - Entorno virtual
- `model/model.bin` - Modelo (477 MB)
- `model/data/` - Modelo distribuido
- `parser/my-languages.so` - Parser compilado
- `parser/vendor/` - Dependencias de tree-sitter
- Cachés y archivos temporales

---

## 📋 Crear README.md para tu Repositorio

```bash
# Crear un README personalizado
cat > README_CUSTOM.md << 'EOF'
# OMPar - Automatic Parallelization with AI

Fork personalizado de OMPar con guías de instalación completas y optimizaciones.

## 🚀 Instalación Rápida

Ver [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) para instrucciones detalladas.

## 📊 Resultados de Pruebas

- **Precisión**: 85.7% (6/7 casos correctos)
- **Sistema**: Ubuntu 24.04, Python 3.12.3, CUDA 13.0
- **Modelos**: OMPify + MonoCoder

## 📝 Guías Incluidas

- [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) - Instalación completa paso a paso
- [cleanup_guide.md](cleanup_guide.md) - Limpieza y optimización de espacio

## ⚙️ Modificaciones

- Compatibilidad con PyTorch 2.5.1 usando safetensors
- Parser compilado con tree-sitter-c-sharp v0.20.0
- Scripts de conversión de modelo incluidos

## 📦 Requisitos

- Python 3.11+
- CUDA 12.1+
- ~10 GB espacio en disco

## 🔗 Repositorio Original

https://github.com/Scientific-Computing-Lab/OMPar

## 📄 Licencia

MIT License
EOF

# Agregar al commit
git add README_CUSTOM.md
git commit -m "Add custom README with installation instructions"
```

---

## 🔄 Workflow de Trabajo

### Hacer Cambios y Actualizar

```bash
# 1. Hacer cambios en archivos
# ...

# 2. Ver qué cambió
git status
git diff

# 3. Agregar cambios
git add archivo1.py archivo2.py
# O agregar todo:
git add .

# 4. Commit
git commit -m "Descripción de los cambios"

# 5. Subir
git push
```

### Crear Ramas para Experimentos

```bash
# Crear rama nueva
git checkout -b feature/nueva-funcionalidad

# Hacer cambios y commits
git add .
git commit -m "Nueva funcionalidad"

# Subir rama
git push -u origin feature/nueva-funcionalidad

# Volver a main
git checkout main
```

---

## 📝 Archivo .gitattributes (Opcional)

Para manejar archivos grandes con Git LFS:

```bash
cat > .gitattributes << 'EOF'
# Git LFS
*.bin filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.so filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
EOF

git add .gitattributes
git commit -m "Add Git LFS configuration"
```

---

## 🎯 Comandos Rápidos

```bash
# Setup inicial
git init
git remote add origin https://github.com/TU-USUARIO/OMPar.git
git add .
git commit -m "Initial commit"
git branch -M main
git push -u origin main

# Workflow diario
git add .
git commit -m "Descripción del cambio"
git push
```

---

## ⚠️ Importante

1. **No subas el modelo** (`model.bin` - 477 MB) - Está en .gitignore
2. **No subas el entorno virtual** (`ompar_env/`) - Está en .gitignore
3. **Documenta cómo descargar el modelo** en tu README
4. **Incluye las guías de instalación** que creamos

---

## 📧 Compartir con Otros

Para que otros usen tu repositorio:

```bash
# Ellos clonan
git clone https://github.com/TU-USUARIO/OMPar.git
cd OMPar

# Siguen INSTALLATION_GUIDE.md
# Descargan modelo de Google Drive
# Ejecutan convert_model.py
```
