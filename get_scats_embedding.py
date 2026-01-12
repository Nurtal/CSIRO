
import os
from PIL import Image
import numpy as np
import torch
import scattering
from sklearn.preprocessing import StandardScaler


def load_img(img_file:str):
    """ """

    # test if img exist
    if not os.path.isfile(img_file):
        print(f"Can't find the img {img_file}")
        return False

    # LOad img & convert en niveau de gris
    img = Image.open(img_file).convert('L')

    # cast img en numpy
    img_array = np.array(img)

    return img_array


def extract_patches(image, patch_size=128, step=None, normalize='log', mask=None):
    """
    Extract overlapping patches from a 2D image (grayscale or single-channel).

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image, shape (H, W).
    patch_size : int
        Size of the patch (pixels).
    step : int or None
        Step for sliding window. If None, step = patch_size // 2.
    normalize : str or None
        'log' for log scaling, 'zscore' for mean/std normalization, None for raw.
    mask : np.ndarray or None
        Boolean mask of shape (H, W). If provided, skip patches where mask==0.

    Returns
    -------
    patches : torch.Tensor
        Shape (N_patches, 1, patch_size, patch_size)
    coords : np.ndarray
        Shape (N_patches, 2) -> (top, left) pixel coordinates.
    """

    H, W = image.shape

    # Default step
    if step is None:
        step = patch_size // 2

    half = patch_size // 2
    patches = []
    coords = []

    # Default mask: all ones
    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    # Sliding window
    for i in range(half, H - half, step):
        for j in range(half, W - half, step):
            # Extract patch
            patch = image[i - half:i + half, j - half:j + half]
            patch_mask = mask[i - half:i + half, j - half:j + half]

            # Skip if masked
            if np.any(patch_mask == 0):
                continue

            # Normalize
            if normalize == 'log':
                patch = np.log(patch + 1e-6)
            elif normalize == 'zscore':
                patch = (patch - patch.mean()) / (patch.std() + 1e-6)

            # Convert to torch tensor and add channel dimension
            patches.append(torch.from_numpy(patch).float())
            coords.append((i - half, j - half))

    # Stack patches
    patches = torch.stack(patches, dim=0)
    coords = np.array(coords)

    return patches, coords



def normalize_scattering_coefficients(S0, S1, S2, epsilon=1e-6):
    """
    Normalise les coefficients de scattering S0, S1, S2 de manière générale,
    indépendamment des valeurs de J et L.

    Args:
        S0 (np.ndarray): Coefficients S0, forme (1, 1).
        S1 (np.ndarray): Coefficients S1, forme (J, L).
        S2 (np.ndarray): Coefficients S2, forme (J, J, L, L).
        epsilon (float): Constante pour éviter la division par zéro.

    Returns:
        dict: Dictionnaire contenant S0_norm, S1_norm, S2_norm.
    """
    # 1. Réduction de S2 : moyenne sur les dimensions spatiales (axes 2 et 3)
    S2_reduced = S2.mean(axis=(2, 3))  # Forme : (J, J)

    # 2. Normalisation de S1
    # Moyenne de S2_reduced sur les orientations pour correspondre à S1
    # S2_reduced a une forme (J, J), on moyenne sur la 2ème dimension pour obtenir (J, 1)
    S2_reduced_for_S1 = S2_reduced.mean(axis=1, keepdims=True)  # Forme : (J, 1)
    S1_norm = S1 / (S2_reduced_for_S1 + epsilon)  # Broadcasting : (J, L) / (J, 1)

    # 3. Normalisation de S0
    # Moyenne globale de S2_reduced
    S2_global = S2_reduced.mean()  # Valeur scalaire
    S0_norm = S0 / (S2_global + epsilon)  # Forme : (1, 1)

    # 4. Normalisation de S2
    # Étendre S2_reduced pour correspondre à S2
    S2_reduced_expanded = S2_reduced[:, :, np.newaxis, np.newaxis]  # Forme : (J, J, 1, 1)
    S2_norm = S2 / (S2_reduced_expanded + epsilon)  # Forme : (J, J, L, L)

    return {
        "S0_norm": S0_norm,
        "S1_norm": S1_norm,
        "S2_norm": S2_norm
    }
    



def run_scattering(dataset, J,L,patch_size):
    """

    """
    # init 
    M = patch_size
    N = patch_size

    st_calc = scattering.Scattering2d(M=M, N=N, J=J, L=L)

    X_list = []  # Liste pour stocker les vecteurs aplatis de chaque image


    for img in dataset:
        # Ajouter une dimension batch si nécessaire
        img = img[np.newaxis, :, :]

        print(img)
        print("-"*42)

        # Calcul des coefficients de scattering
        S = st_calc.scattering_coef(img)

        # Récupérer S0, S1, S2 et les convertir en numpy
        S0 = S["S0"][0].cpu().numpy()
        S1 = S["S1"][0].cpu().numpy() 
        S2 = S["S2"][0].cpu().numpy() 

        # Normalisation
        epsilon = 1e-6
    
        normalized_coeffs = normalize_scattering_coefficients(S0, S1, S2)
        
        # Aplatir et concaténer les coefficients normalisés
        S0_flat = normalized_coeffs['S0_norm'].flatten() 
        S1_flat = normalized_coeffs['S1_norm'].flatten()
        S2_flat = normalized_coeffs['S2_norm'].flatten() 
        X_flat = np.concatenate([S0_flat, S1_flat, S2_flat]) 

        # Ajouter au tableau global
        X_list.append(X_flat)

        # Convertir la liste en tableau 2D
        X = np.vstack(X_list) 

    return X




def run_scattering_single_shot(img, J,L):
    """Compute scat (no patches, one vector for one image)

    """
    # init 
    h, w = img.shape
    st_calc = scattering.Scattering2d(M=h, N=w, J=J, L=L)

    # black magic (scat lib attends un tenseur torch)
    img = torch.from_numpy(img).float()
    
    # Ajouter une dimension batch si nécessaire
    img = img[np.newaxis, :, :]

    # Calcul des coefficients de scattering
    S = st_calc.scattering_coef(img)

    # Récupérer S0, S1, S2 et les convertir en numpy
    S0 = S["S0"][0].cpu().numpy()
    S1 = S["S1"][0].cpu().numpy() 
    S2 = S["S2"][0].cpu().numpy() 

    # Normalisation
    epsilon = 1e-6
    normalized_coeffs = normalize_scattering_coefficients(S0, S1, S2)

    # Aplatir et concaténer les coefficients normalisés
    S0_flat = normalized_coeffs['S0_norm'].flatten() 
    S1_flat = normalized_coeffs['S1_norm'].flatten()
    S2_flat = normalized_coeffs['S2_norm'].flatten() 
    X = np.concatenate([S0_flat, S1_flat, S2_flat]) 

    return X


def compute_embedding(image_name:str, j:int, l:int):
    """ """

    # load image
    image = load_img(image_name)
    
    # run scats
    X = run_scattering_single_shot(image, j,l)

    # 'normalize' data
    X = np.nan_to_num(X)

    return X



if __name__ == "__main__":

    # params
    img_file = "data/biomass/train/ID1011485656.jpg"
    j = 6
    l = 1
    patch_size = 32

    # run
    v = compute_embedding(img_file, j, l, patch_size)

    print(v)
    
