#!/usr/bin/env python3
"""
Skrypt do zarządzania archiwami modeli OCR.

Użycie:
    python manage_models.py list              - Wylistuj wszystkie archiwa
    python manage_models.py cleanup [N]       - Usuń starsze modele, zachowaj N ostatnich
    python manage_models.py archive <path>    - Zarchiwizuj model
    python manage_models.py restore <archive> <target> - Przywróć model z archiwum
"""

import sys
from pathlib import Path

# Dodaj katalog ocr do ścieżki importu
sys.path.insert(0, str(Path(__file__).parent))

from ocr import info
from ocr.model_archive import ModelArchiver
from ocr.config import MODEL_ARCHIVE_DIR, MODEL_ARCHIVE_KEEP_COUNT


def print_usage():
    """Wyświetla instrukcję użycia."""
    info(__doc__)


def list_archives():
    """Wylistowuje wszystkie archiwa modeli."""
    archiver = ModelArchiver(MODEL_ARCHIVE_DIR)
    archiver.print_archives_summary()


def cleanup_archives(keep_count: int = None):
    """Czyści stare archiwa."""
    if keep_count is None:
        keep_count = MODEL_ARCHIVE_KEEP_COUNT
    
    archiver = ModelArchiver(MODEL_ARCHIVE_DIR)
    deleted = archiver.cleanup_old_archives(keep_count=keep_count)
    info(f"\n✓ Operacja zakończona. Usunięto {deleted} archiwów.")


def archive_model(model_path: str, accuracy: float = 0.0, epoch: int = 0):
    """Archiwizuje model."""
    archiver = ModelArchiver(MODEL_ARCHIVE_DIR)
    result = archiver.archive_model(model_path, accuracy=accuracy, epoch=epoch)
    
    if result:
        info(f"\n✓ Model zarchiwizowany pomyślnie: {result}")
    else:
        info(f"\n✗ Błąd podczas archiwizacji modelu")
        sys.exit(1)


def restore_model(archive_filename: str, target_path: str):
    """Przywraca model z archiwum."""
    archiver = ModelArchiver(MODEL_ARCHIVE_DIR)
    success = archiver.restore_model(archive_filename, target_path)
    
    if success:
        info(f"\n✓ Model przywrócony pomyślnie")
    else:
        info(f"\n✗ Błąd podczas przywracania modelu")
        sys.exit(1)


def main():
    """Główna funkcja."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_archives()
    
    elif command == "cleanup":
        keep_count = int(sys.argv[2]) if len(sys.argv) > 2 else MODEL_ARCHIVE_KEEP_COUNT
        cleanup_archives(keep_count)
    
    elif command == "archive":
        if len(sys.argv) < 3:
            info("Błąd: podaj ścieżkę do modelu")
            info("Użycie: python manage_models.py archive <path> [accuracy] [epoch]")
            sys.exit(1)
        
        model_path = sys.argv[2]
        accuracy = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        epoch = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        archive_model(model_path, accuracy, epoch)
    
    elif command == "restore":
        if len(sys.argv) < 4:
            info("Błąd: podaj archiwum i ścieżkę docelową")
            info("Użycie: python manage_models.py restore <archive_filename> <target_path>")
            sys.exit(1)
        
        archive_filename = sys.argv[2]
        target_path = sys.argv[3]
        restore_model(archive_filename, target_path)
    
    else:
        info(f"Nieznana komenda: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
