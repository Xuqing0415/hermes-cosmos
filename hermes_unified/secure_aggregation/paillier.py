"""
Paillier Homomorphic Encryption for Secure Aggregation

Implements Paillier additive homomorphic encryption.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import random

logger = logging.getLogger(__name__)


class PaillierKeyPair:
    """
    Paillier key pair generator.
    
    Generates public/private key pair for Paillier encryption.
    """
    
    def __init__(self, key_size: int = 1024):
        self.key_size = key_size
        self.public_key, self.private_key = self._generate_keypair()
        
    def _generate_keypair(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Generate Paillier key pair.
        
        For demonstration, we use small primes.
        In production, use secure key generation!
        """
        # Small primes for demonstration purposes
        p = self._generate_prime(self.key_size // 2)
        q = self._generate_prime(self.key_size // 2)
        
        n = p * q
        g = n + 1  # g = n+1 is common choice
        lambda_ = (p - 1) * (q - 1)
        mu = self._modular_inverse(lambda_, n)
        
        public_key = (n, g)
        private_key = (lambda_, mu)
        
        return public_key, private_key
    
    def _generate_prime(self, bits: int) -> int:
        """Generate a small prime number for demonstration."""
        candidate = random.randint(2**(bits-1), 2**bits - 1)
        # Simple primality check for demo
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for p in small_primes:
            if candidate % p == 0:
                candidate += 1
        return candidate
    
    def _modular_inverse(self, a: int, m: int) -> int:
        """Compute modular inverse using extended Euclidean algorithm."""
        g, x, _ = self._extended_gcd(a, m)
        if g != 1:
            raise ValueError("Modular inverse does not exist")
        return x % m
    
    def _extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean algorithm."""
        if a == 0:
            return (b, 0, 1)
        g, y, x = self._extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)


class PaillierEncryption:
    """
    Paillier additive homomorphic encryption.
    
    Supports:
    - Encryption
    - Decryption
    - Homomorphic addition
    - Homomorphic scalar multiplication
    """
    
    def __init__(self, public_key: Tuple[int, int], 
                 private_key: Optional[Tuple[int, int]] = None):
        """
        Initialize Paillier encryption.
        
        Args:
            public_key: (n, g)
            private_key: (lambda, mu), optional (only needed for decryption)
        """
        self.n, self.g = public_key
        self.private_key = private_key
        
    def encrypt(self, m: int) -> int:
        """
        Encrypt a message.
        
        Args:
            m: Plaintext integer
        
        Returns:
            Ciphertext
        """
        # Ensure message is within range
        m = m % self.n
        
        # Choose random r
        r = random.randint(1, self.n - 1)
        
        # Compute: c = g^m * r^n mod n^2
        n_sq = self.n * self.n
        c = (pow(self.g, m, n_sq) * pow(r, self.n, n_sq)) % n_sq
        
        return c
    
    def decrypt(self, c: int) -> int:
        """
        Decrypt a ciphertext.
        
        Args:
            c: Ciphertext
        
        Returns:
            Plaintext integer
        """
        if self.private_key is None:
            raise ValueError("Private key needed for decryption")
        
        lambda_, mu = self.private_key
        n_sq = self.n * self.n
        
        # Compute: L(c^lambda mod n^2) where L(x) = (x-1)/n
        c_lambda = pow(c, lambda_, n_sq)
        l = (c_lambda - 1) // self.n
        
        # Plaintext: l * mu mod n
        m = (l * mu) % self.n
        
        return m
    
    def add_encrypted(self, c1: int, c2: int) -> int:
        """
        Homomorphic addition: E(m1) + E(m2) = E(m1 + m2)
        
        Args:
            c1: Encryption of m1
            c2: Encryption of m2
        
        Returns:
            Encryption of m1 + m2
        """
        n_sq = self.n * self.n
        return (c1 * c2) % n_sq
    
    def multiply_by_scalar(self, c: int, scalar: int) -> int:
        """
        Homomorphic scalar multiplication: E(m) * k = E(k*m)
        
        Args:
            c: Encryption of m
            scalar: Scalar k
        
        Returns:
            Encryption of k*m
        """
        n_sq = self.n * self.n
        return pow(c, scalar, n_sq)


class PaillierArrayEncryption:
    """
    Paillier encryption for numpy arrays.
    """
    
    def __init__(self, public_key: Tuple[int, int], 
                 private_key: Optional[Tuple[int, int]] = None,
                 scale_factor: int = 1000):
        """
        Initialize array encryption.
        
        Args:
            public_key: Paillier public key
            private_key: Optional private key
            scale_factor: Scale factor for floating points
        """
        self.encryptor = PaillierEncryption(public_key, private_key)
        self.scale_factor = scale_factor
        
    def encrypt_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Encrypt a numpy array.
        
        Args:
            arr: Array to encrypt (floating point)
        
        Returns:
            Encrypted array as integers
        """
        # Scale to integers
        arr_scaled = (arr * self.scale_factor).astype(np.int64)
        
        encrypted = np.zeros_like(arr_scaled, dtype=np.int64)
        
        for idx in np.ndindex(arr_scaled.shape):
            encrypted[idx] = self.encryptor.encrypt(int(arr_scaled[idx]))
        
        return encrypted
    
    def decrypt_array(self, encrypted_arr: np.ndarray) -> np.ndarray:
        """
        Decrypt an encrypted array.
        
        Args:
            encrypted_arr: Encrypted array
        
        Returns:
            Decrypted array
        """
        decrypted = np.zeros_like(encrypted_arr, dtype=np.float64)
        
        for idx in np.ndindex(encrypted_arr.shape):
            decrypted_val = self.encryptor.decrypt(int(encrypted_arr[idx]))
            decrypted[idx] = decrypted_val / self.scale_factor
        
        return decrypted
    
    def add_encrypted_arrays(self, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """
        Add two encrypted arrays homomorphically.
        """
        result = np.zeros_like(c1, dtype=np.int64)
        
        for idx in np.ndindex(c1.shape):
            result[idx] = self.encryptor.add_encrypted(int(c1[idx]), int(c2[idx]))
        
        return result
    
    def multiply_by_scalar_array(self, c: np.ndarray, scalar: int) -> np.ndarray:
        """
        Multiply encrypted array by scalar homomorphically.
        """
        result = np.zeros_like(c, dtype=np.int64)
        
        for idx in np.ndindex(c.shape):
            result[idx] = self.encryptor.multiply_by_scalar(int(c[idx]), scalar)
        
        return result


# For demonstration purposes, create a simple insecure version
def create_demo_paillier() -> Tuple[PaillierEncryption, PaillierArrayEncryption]:
    """Create demo Paillier encryption (small keys, insecure)."""
    keypair = PaillierKeyPair(key_size=128)
    paillier = PaillierEncryption(keypair.public_key, keypair.private_key)
    array_paillier = PaillierArrayEncryption(keypair.public_key, keypair.private_key)
    return paillier, array_paillier
