import os
import pydicom
import numpy as np
import torch
import torch.nn.functional as F
import random
from pathlib import Path
from tqdm import tqdm

# --- Configurações ---
# Garanta que o caminho para o seu dataset está correto
dataset_dir = "/mnt/HD_1/dados/dadosFazekas/tmp_20"
label_mapping = {"0.5": 0.0, "1.5": 1.0, "2.5": 2.0}
TARGET_SHAPE_2D = (256, 256) # Tamanho alvo 2D para as fatias
MODALITY = "dicom_fazekas_unimodal"


def preprocess_dicom_minimal(dicom_path):
    """
    Carrega o DICOM e retorna o tensor de imagem e os dados.
    A imagem é escalada provisoriamente para 0-1 antes da normalização final por percentil.
    """
    try:
        dicom_data = pydicom.dcmread(dicom_path, force=True)
        image = dicom_data.pixel_array.astype(np.float32)
        # Escala inicial 0-1 (melhora interpolação)
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        image = torch.tensor(image).unsqueeze(0)  # C=1: (1, H, W)
        return image, dicom_data
    except Exception as e:
        # print(f"Erro ao processar DICOM {dicom_path}: {e}")
        return None, None

def extract_label(dicom_data):
    """Extrai a label Fazekas customizada."""
    tag_value = dicom_data.get((0x0011, 0x1001),)
    if hasattr(tag_value, 'value'):
        return str(tag_value.value)
    return str(tag_value)


def normalise_percentile(volume):
    """
    Normaliza o volume 5D (1, C, H, W, D) pela escala do 99º percentil do foreground (nonzero).
    Adaptado do Segundo Código para a forma 5D (1, 1, H, W, D).
    """
    # volume.shape[1] é o número de modalidades/canais (que é 1 neste caso)
    for mdl in range(volume.shape[1]): # Loop para o canal de modalidade
        # Extrai o volume 3D (H, W, D) para a modalidade 'mdl'
        v_3d = volume[0, mdl, :, :, :]
        # Acha o array 1D de todos os pixels > 0 (foreground)
        v_ = v_3d.reshape(-1)
        v_ = v_[v_ > 0]  # Usa apenas o foreground (não-zero)
        
        if v_.numel() == 0:
            continue
            
        p_99 = torch.quantile(v_, 0.99)
        
        # Adaptando para a forma 5D:
        if p_99 > 1e-6:
            volume[0, mdl, :, :, :] /= p_99
        else:
            volume[0, mdl, :, :, :] = torch.zeros_like(volume[0, mdl, :, :, :])
            
    return volume

def process_patient_series_slices(serie_path, target_path, final_label):
    """
    Processa uma série DICOM, aplica normalização, filtra fatias com cérebro e salva as fatias 2D (X e Y).
    """
    dicom_files = sorted([f for f in os.listdir(serie_path) if f.endswith(".dcm")])
    if not dicom_files:
        return None
        
    series_images = []
    
    # 1. Pré-processa e redimensiona cada fatia
    for dicom_file in dicom_files:
        dicom_path = os.path.join(serie_path, dicom_file)
        image_tensor, dicom_data = preprocess_dicom_minimal(dicom_path) # image_tensor: (1, H, W)
        
        if image_tensor is None:
            continue

        # Redimensiona para TARGET_SHAPE_2D (256x256)
        # Usa .unsqueeze(0) para (1, 1, H, W), então redimensiona (modo bilinear), e remove o batch dim (.squeeze(0))
        image_resized = F.interpolate(image_tensor.unsqueeze(0), size=TARGET_SHAPE_2D, mode='bilinear', align_corners=False).squeeze(0)
        series_images.append(image_resized)
        
    if not series_images:
        return None

    # 2. Empilha e ajusta a dimensão para (1, 1, H, W, D)
    # series_images[i] é (1, H, W). Stack no dim=1 resulta em (1, num_slices, H, W)
    volume_stacked = torch.stack(series_images, dim=1)
    # Permute para a ordem (1, C, H, W, D) = (1, 1, H, W, num_slices)
    volume = volume_stacked.unsqueeze(0).permute(0, 1, 3, 4, 2)
    
    # 3. Normalização do Volume (99º Percentil)
    volume = normalise_percentile(volume)
    
    # 4. Filtragem de Fatias Vazias (Lógica do Segundo Código)
    # volume.shape: (1, 1, H, W, D)
    
    # Média dos canais (ainda 1 canal) -> (H, W, D). volume[0] é (1, H, W, D)
    mean_volume_slice = volume[0].mean(dim=0) # (H, W, D)
    
    # Soma (H, W) para cada fatia D; Usa 0.5 para considerar valores após normalização
    # Sum(axis=0).sum(axis=0) soma H e W, resultando em (D)
    sum_dim_slices = (mean_volume_slice.sum(axis=0).sum(axis=0) > 0.5).int()
    
    num_slices = volume.shape[-1]
    
    # Encontra o primeiro slice (fs_dim): primeiro índice > 0
    fs_dim = sum_dim_slices.argmax()
    # Encontra o último slice (ls_dim): num_slices - (primeiro índice > 0 da array invertida)
    # Se todos forem > 0, o argmax da flip é 0, e ls_dim = num_slices - 0 = num_slices. Correto.
    ls_dim = num_slices - sum_dim_slices.flip(dims=[0]).argmax()
    
    print(f"Série {Path(serie_path).name} tem {num_slices} fatias. Processando de {fs_dim} até {ls_dim}.")
    
    if fs_dim == ls_dim:
         print("Aviso: Nenhuma fatia com conteúdo cerebral encontrada após o filtro.", flush=True)
         return None

    # 5. Salvamento das Fatias (Incluindo Máscara Y de Zeros)
    patient_dir = target_path / f"serie_{Path(serie_path).name}"
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva a label final do volume (Fazekas)
    with open(patient_dir / "label.txt", "w") as f:
        f.write(str(final_label))
        
    # Gera a máscara de lesão (Y) - Zero Mask, pois Fazekas é label de volume
    zero_mask = torch.zeros((1, 1) + TARGET_SHAPE_2D, dtype=torch.float32)

    # Itera apenas sobre as fatias com conteúdo cerebral
    for slice_idx in range(fs_dim, ls_dim):
        # volume[:, :, :, :, slice_idx] -> (1, 1, H, W)
        slice_data_x = volume[:, :, :, :, slice_idx] 
        
        # Salva a imagem X e a máscara Y (zero mask)
        # O AnoFPDM precisa dessa estrutura X e Y para carregar o dataset, mesmo que Y seja zero.
        np.savez_compressed(
            patient_dir / f"slice_{slice_idx}", 
            x=slice_data_x.numpy(), # (1, 1, H, W)
            y=zero_mask.numpy()     # (1, 1, H, W)
        )

    return True # Retorna True se processado

def determine_label(serie_path):
    """Apenas determina a label da série."""
    dicom_files = sorted([f for f in os.listdir(serie_path) if f.endswith(".dcm")])
    if not dicom_files:
        return -1
    
    # Usa a primeira fatia para extrair a label
    dicom_path = os.path.join(serie_path, dicom_files[0])
    dicom_data = pydicom.dcmread(dicom_path, force=True)
    
    label_value_str = extract_label(dicom_data)
    final_label = label_mapping.get(label_value_str, label_value_str)
    
    try:
        # Padroniza F2 e F3 como 2 (Anomalia Severa)
        if float(final_label) >= 2.0:
            final_label = 2
        # F1 como 1 (Anomalia Leve)
        elif float(final_label) == 1.0:
            final_label = 1
        # F0 como 0 (Normal - DOMÍNIO DE TREINO)
        else:
            final_label = 0
            
        return int(final_label)
    except:
        return -1 # Label inválida

def preprocess(datapath: Path, shape=256):

    # 1. Encontrar todos os caminhos de séries
    all_series_paths = []
    for root, dirs, files in os.walk(datapath):
        if any(f.endswith(".dcm") for f in files):
            all_series_paths.append(Path(root))
    all_series_paths = sorted(all_series_paths)
    
    base_preprocess_dir = Path("./preprocess")  # Pasta fixa para salvar os dados pré-processados
    splits_path = base_preprocess_dir / "data_splits"

    
    # --- Lógica de Split (Treino APENAS com Normais - AnoFPDM) ---
    if not splits_path.exists():
        
        # 2. Apurar labels para separar Normal/Anômalo
        series_labels = {}
        for serie_path in tqdm(all_series_paths, desc="Apurando labels para Split"):
            series_labels[serie_path] = determine_label(serie_path)
            
        # Filtra séries com labels válidas (0, 1, 2)
        valid_paths = [p for p, l in series_labels.items() if l in (0, 1, 2)]
        normal_paths = [p for p, l in series_labels.items() if l == 0]
        anomalous_paths = [p for p, l in series_labels.items() if l in (1, 2)]
        
        # Embaralhamento
        random.seed(10)
        random.shuffle(normal_paths)
        random.shuffle(anomalous_paths)
        
        # 3. Gerar Splits
        
        # 70% de Normais para Treino (Domínio 'Normal')
        n_train_normal = int(len(normal_paths) * 0.7)
        train_paths = normal_paths[:n_train_normal]
        
        # O restante (30% Normais + Todos os Anômalos) vai para Validação e Teste
        test_val_paths = normal_paths[n_train_normal:] + anomalous_paths
        
        # Distribuição de Validação/Teste (50%/50% do restante)
        n_val = int(len(test_val_paths) * 0.5) 
        
        val_paths = test_val_paths[:n_val]
        test_paths = test_val_paths[n_val:]
        
        # 4. Salvar os Splits (apenas os caminhos relativos)
        split_paths = {"train": train_paths, "val": val_paths, "test": test_paths}
        
        for split in ["train", "val", "test"]:
            (splits_path / split).mkdir(parents=True, exist_ok=True)
            with open(splits_path / split / "scans.csv", "w") as f:
                # Salva o caminho relativo da série
                f.write("\n".join([str(p.relative_to(datapath)) for p in split_paths[split]]))
                
        print(f"Split de Treino (Normal): {len(train_paths)} séries")
        print(f"Split de Validação (Normal/Anômalo): {len(val_paths)} séries")
        print(f"Split de Teste (Normal/Anômalo): {len(test_paths)} séries")
        
    # --- Processamento e Salvamento de Fatias ---
    
    for split in ["train", "val", "test"]:
        with open(splits_path / split / "scans.csv") as f:
            # Reconstroi o caminho completo
            paths = [datapath / x.strip() for x in f.readlines()]

        print(f"\nIniciando Processamento de Fatias em [{split}]: {len(paths)} séries")

        target_path = base_preprocess_dir / f"npy_{split}"
        target_path.mkdir(parents=True, exist_ok=True)


        for source_path in tqdm(paths):
            # A label é determinada para ser salva no arquivo label.txt
            final_label = determine_label(source_path)
            
            # Garante que pacientes inválidos (label -1) não sejam processados/salvos
            if final_label == -1:
                continue

            process_patient_series_slices(source_path, target_path, final_label)


if __name__ == "__main__":
    
    datapath = Path(dataset_dir)
    
    print(f"Iniciando pré-processamento de fatias Fazekas para {MODALITY}...")
    
    preprocess(datapath)