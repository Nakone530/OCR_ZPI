"""
Moduł do archiwizacji poprzednich modeli OCR.

Odpowiedzialności:
  - Archiwizowanie starych modeli do katalogu archiwum
  - Czyszczenie poprzednich wersji modelu
  - Utrzymywanie historii modeli z datami i statystykami
  - Zarządzanie przestrzenią dyskową
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class ModelArchiver:
    """Klasa do zarządzania archiwizacją modeli OCR."""
    
    def __init__(self, archive_dir: str = "./model_archive"):
        """
        Inicjalizuje archiwiście modeli.
        
        Args:
            archive_dir: Katalog do przechowywania archiwów modeli
        """
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.archive_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    
    def _load_manifest(self) -> Dict:
        """Wczytuje manifest archiwów z dysku."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[WARN] Nie udało się wczytać manifestu: {e}")
                return {"archives": []}
        return {"archives": []}
    
    
    def _save_manifest(self) -> None:
        """Zapisuje manifest archiwów na dysk."""
        try:
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[ERROR] Nie udało się zapisać manifestu: {e}")
    
    
    def archive_model(
        self,
        model_path: str,
        accuracy: float = 0.0,
        epoch: int = 0,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Archiwizuje model do katalogu archiwum.
        
        Args:
            model_path: Ścieżka do modelu do archiwizacji
            accuracy: Dokładność modelu (%)
            epoch: Liczba epoki treningu
            tags: Lista tagów/etykiet dla archiwum (np. ["best", "checkpoint"])
        
        Returns:
            Ścieżka do zarchiwizowanego modelu lub None jeśli błąd
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"[ERROR] Model nie istnieje: {model_path}")
            return None
        
        # Generuj unikalną nazwę archiwum z datą i czasem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_filename = f"model_{timestamp}.pth"
        archived_path = self.archive_dir / archived_filename
        
        try:
            # Kopiuj model do archiwum
            shutil.copy2(model_path, archived_path)
            print(f"[ARCHIVE] Model zarchiwizowany: {archived_path}")
            
            # Aktualizuj manifest
            archive_entry = {
                "filename": archived_filename,
                "original_path": str(model_path),
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "accuracy": float(accuracy),
                "epoch": int(epoch),
                "tags": tags or [],
                "size_bytes": archived_path.stat().st_size
            }
            
            self.manifest["archives"].append(archive_entry)
            self._save_manifest()
            
            return str(archived_path)
            
        except (IOError, OSError) as e:
            print(f"[ERROR] Błąd podczas archiwizacji: {e}")
            return None
    
    
    def cleanup_old_archives(self, keep_count: int = 10) -> int:
        """
        Usuwa starsze archiwa, pozostawiając tylko ostatnie N.
        
        Args:
            keep_count: Liczba ostatnich archiwów do zachowania
        
        Returns:
            Liczba usuniętych archiwów
        """
        if len(self.manifest["archives"]) <= keep_count:
            print(f"[INFO] Brak starych archiwów do usunięcia (mamy {len(self.manifest['archives'])}, zachowujemy {keep_count})")
            return 0
        
        # Sortuj archiwa po dacie (najnowsze na końcu)
        sorted_archives = sorted(
            self.manifest["archives"],
            key=lambda x: x["datetime"]
        )
        
        # Określ które archiwa należy usunąć
        archives_to_delete = sorted_archives[:-keep_count]
        deleted_count = 0
        
        for archive in archives_to_delete:
            archive_path = self.archive_dir / archive["filename"]
            try:
                if archive_path.exists():
                    archive_path.unlink()
                    print(f"[CLEANUP] Usunięto: {archive['filename']}")
                    deleted_count += 1
                
                # Usuń z manifestu
                self.manifest["archives"].remove(archive)
                
            except OSError as e:
                print(f"[ERROR] Nie udało się usunąć {archive_path}: {e}")
        
        if deleted_count > 0:
            self._save_manifest()
            print(f"[CLEANUP] Usunięto {deleted_count} starych archiwów")
        
        return deleted_count
    
    
    def list_archives(self) -> List[Dict]:
        """
        Wylistowuje wszystkie zarchiwizowane modele.
        
        Returns:
            Lista archiwów z ich metadanymi
        """
        # Sortuj malejąco po dacie (najnowsze na początku)
        sorted_archives = sorted(
            self.manifest["archives"],
            key=lambda x: x["datetime"],
            reverse=True
        )
        return sorted_archives
    
    
    def print_archives_summary(self) -> None:
        """Drukuje podsumowanie zarchiwizowanych modeli."""
        archives = self.list_archives()
        
        if not archives:
            print("[INFO] Brak zarchiwizowanych modeli")
            return
        
        print("\n" + "="*80)
        print("  ZARCHIWIZOWANE MODELE")
        print("="*80)
        
        for idx, archive in enumerate(archives, 1):
            size_mb = archive["size_bytes"] / (1024 * 1024)
            tags_str = ", ".join(archive.get("tags", [])) or "brak"
            
            print(f"\n{idx}. {archive['filename']}")
            print(f"   Data: {archive['datetime']}")
            print(f"   Dokładność: {archive['accuracy']:.2f}%")
            print(f"   Epoka: {archive['epoch']}")
            print(f"   Tagi: {tags_str}")
            print(f"   Rozmiar: {size_mb:.2f} MB")
        
        print("\n" + "="*80)
    
    
    def restore_model(self, archive_filename: str, target_path: str) -> bool:
        """
        Przywraca model z archiwum.
        
        Args:
            archive_filename: Nazwa pliku w archiwum (np. model_20260330_120000.pth)
            target_path: Ścieżka docelowa dla przywróconego modelu
        
        Returns:
            True jeśli powodzenie, False jeśli błąd
        """
        archive_path = self.archive_dir / archive_filename
        
        if not archive_path.exists():
            print(f"[ERROR] Archiwum nie istnieje: {archive_path}")
            return False
        
        try:
            shutil.copy2(archive_path, target_path)
            print(f"[RESTORE] Model przywrócony z: {archive_filename}")
            print(f"[RESTORE] Zapisano do: {target_path}")
            return True
        except (IOError, OSError) as e:
            print(f"[ERROR] Błąd podczas przywracania: {e}")
            return False
    
    
    def get_archive_info(self, archive_filename: str) -> Optional[Dict]:
        """
        Pobiera informacje o konkretnym archiwum.
        
        Args:
            archive_filename: Nazwa pliku archiwum
        
        Returns:
            Słownik z informacjami archiwum lub None
        """
        for archive in self.manifest["archives"]:
            if archive["filename"] == archive_filename:
                return archive
        return None


def archive_model_simple(
    model_path: str,
    accuracy: float = 0.0,
    epoch: int = 0,
    archive_dir: str = "./model_archive"
) -> Optional[str]:
    """
    Archiwizuje model - wersja uproszczona dla szybkiego użytku.
    
    Args:
        model_path: Ścieżka do modelu do archiwizacji
        accuracy: Dokładność modelu (%)
        epoch: Liczba epoki treningu
        archive_dir: Katalog archiwum
    
    Returns:
        Ścieżka do zarchiwizowanego modelu
    """
    archiver = ModelArchiver(archive_dir)
    return archiver.archive_model(model_path, accuracy, epoch)


def cleanup_old_models(keep_count: int = 10, archive_dir: str = "./model_archive") -> int:
    """
    Usuwa stare modele, pozostawiając ostatnie N.
    
    Args:
        keep_count: Liczba ostatnich modeli do zachowania
        archive_dir: Katalog archiwum
    
    Returns:
        Liczba usuniętych modeli
    """
    archiver = ModelArchiver(archive_dir)
    return archiver.cleanup_old_archives(keep_count)
