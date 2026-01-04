import torch
import argparse
from compAI import OMPAR

# Configurar argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='tokenizer/gpt/gpt_vocab/gpt2-vocab.json')
parser.add_argument('--merge_file', default='tokenizer/gpt/gpt_vocab/gpt2-merges.txt')
parser.add_argument('--model_weights', default='model')
args = parser.parse_args([])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'🚀 Iniciando pruebas simples de OMPar')
print(f'Dispositivo: {device}\n')

ompar = OMPAR(model_path=args.model_weights, device=device, args=args)

# Lista de pruebas
tests = [
    {
        'name': 'Inicialización de array',
        'code': '''for (int i = 0; i < n; i++) {
    arr[i] = 0;
}''',
        'expected': 'Paralelizable'
    },
    {
        'name': 'Suma acumulativa (reducción)',
        'code': '''for (int i = 0; i < n; i++) {
    total += arr[i];
}''',
        'expected': 'Paralelizable con reduction'
    },
    {
        'name': 'Copia de array',
        'code': '''for (int i = 0; i < n; i++) {
    dest[i] = src[i];
}''',
        'expected': 'Paralelizable'
    },
    {
        'name': 'Operación elemento a elemento',
        'code': '''for (int i = 0; i < n; i++) {
    result[i] = a[i] + b[i] * c[i];
}''',
        'expected': 'Paralelizable'
    },
    {
        'name': 'Búsqueda con break',
        'code': '''for (int i = 0; i < n; i++) {
    if (arr[i] == target) {
        break;
    }
}''',
        'expected': 'NO paralelizable'
    },
    {
        'name': 'Fibonacci (dependencia)',
        'code': '''for (int i = 2; i < n; i++) {
    fib[i] = fib[i-1] + fib[i-2];
}''',
        'expected': 'NO paralelizable'
    },
    {
        'name': 'Máximo (reducción)',
        'code': '''for (int i = 0; i < n; i++) {
    if (arr[i] > max) max = arr[i];
}''',
        'expected': 'Paralelizable con reduction'
    },
    {
        'name': 'Normalización',
        'code': '''for (int i = 0; i < n; i++) {
    arr[i] = arr[i] / sum;
}''',
        'expected': 'Paralelizable'
    }
]

# Ejecutar pruebas
print('=' * 80)
for i, test in enumerate(tests, 1):
    print(f'\n📝 Prueba {i}: {test["name"]}')
    print(f'Esperado: {test["expected"]}')
    print('-' * 80)
    print(f'Código:\n{test["code"]}')
    print('-' * 80)
    
    pragma = ompar.auto_comp(test['code'])
    
    if pragma:
        print(f'✅ Resultado: #pragma {pragma}')
    else:
        print(f'❌ Resultado: No paralelizable')
    
    print('=' * 80)

print('\n✅ Pruebas completadas!')
