#!/usr/bin/env python3
"""
Setup script for Barcelona Twitter Bot
Installs dependencies, sets up Ollama, creates directories and .env
"""
import os
import sys
import subprocess
import platform
import shutil
import time
import urllib.request
from pathlib import Path

# Minimal console output before rich is installed
def print_info(msg): print(f"  -> {msg}")
def print_ok(msg): print(f"  [OK] {msg}")
def print_warn(msg): print(f"  [WARN] {msg}")
def print_err(msg): print(f"  [ERROR] {msg}")
def print_step(msg): print(f"\n=== {msg} ===")

PROJECT_ROOT = Path(__file__).parent
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
ENV_FILE = PROJECT_ROOT / ".env"


def check_python_version():
    """Check Python version >= 3.10"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_err(f"Python 3.10+ requis. Version actuelle: {version.major}.{version.minor}.{version.micro}")
        return False
    print_ok(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_venv():
    """Create virtual environment"""
    if VENV_DIR.exists():
        print_warn("Environnement virtuel déjà existant, réinstallation des dépendances...")
        return True

    print_info("Création de l'environnement virtuel...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True, capture_output=True)
        print_ok("Environnement virtuel créé")
        return True
    except subprocess.CalledProcessError as e:
        print_err(f"Erreur création venv: {e}")
        return False


def get_pip_path():
    """Get pip executable path for current platform"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def get_python_path():
    """Get python executable path for current platform"""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_dependencies():
    """Install Python dependencies"""
    pip_path = get_pip_path()
    if not pip_path.exists():
        print_err("pip non trouvé dans le venv")
        return False

    print_info("Installation des dépendances Python (peut prendre 1-2 min)...")
    try:
        # Install requirements directly (skip pip upgrade to avoid issues)
        result = subprocess.run([str(pip_path), "install", "-r", str(REQUIREMENTS)], check=True, capture_output=True, text=True, timeout=300)
        print_ok("Dépendances installées")
        return True
    except subprocess.CalledProcessError as e:
        print_err(f"Erreur installation: {e.stderr[:500]}")
        return False
    except subprocess.TimeoutExpired:
        print_err("Timeout installation dépendances (>5 min)")
        return False


def init_rich():
    """Initialize rich after dependencies are installed - runs in venv Python"""
    python_path = get_python_path()
    test_code = """
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
print("RICH_OK")
"""
    try:
        result = subprocess.run([str(python_path), "-c", test_code], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and "RICH_OK" in result.stdout:
            # Now import in main process
            from rich.console import Console
            from rich.panel import Panel
            from rich.progress import Progress, SpinnerColumn, TextColumn
            from rich.prompt import Confirm
            globals()['console'] = Console()
            globals()['Panel'] = Panel
            globals()['Progress'] = Progress
            globals()['SpinnerColumn'] = SpinnerColumn
            globals()['TextColumn'] = TextColumn
            globals()['Confirm'] = Confirm
            return True
    except Exception as e:
        print_err(f"Impossible d'importer rich: {e}")
    return False


def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(["ollama", "version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            if console:
                console.print(f"[green]✅ Ollama installé: {result.stdout.strip()}[/green]")
            else:
                print_ok(f"Ollama installé: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def install_ollama_windows():
    """Install Ollama on Windows"""
    if console:
        console.print("[cyan]📥 Téléchargement et installation d'Ollama...[/cyan]")
    else:
        print_info("Téléchargement et installation d'Ollama...")

    try:
        installer_url = "https://ollama.com/download/OllamaSetup.exe"
        installer_path = PROJECT_ROOT / "OllamaSetup.exe"

        urllib.request.urlretrieve(installer_url, installer_path)
        subprocess.run([str(installer_path), "/S"], check=True, timeout=120)
        installer_path.unlink(missing_ok=True)

        if console:
            console.print("[green]✅ Ollama installé[/green]")
        else:
            print_ok("Ollama installé")
        return True
    except Exception as e:
        if console:
            console.print(f"[red]❌ Erreur installation Ollama: {e}[/red]")
            console.print("[yellow]💡 Installez manuellement: https://ollama.com/download[/yellow]")
        else:
            print_err(f"Erreur installation Ollama: {e}")
            print_warn("Installez manuellement: https://ollama.com/download")
        return False


def pull_ollama_model(model="qwen2.5:7b"):
    """Pull the required Ollama model"""
    if console:
        console.print(f"[cyan]📥 Téléchargement du modèle {model} (peut prendre quelques minutes)...[/cyan]")
    else:
        print_info(f"Téléchargement du modèle {model}...")

    try:
        if console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Pull {model}...", total=None)
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
        else:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=600
            )

        if result.returncode == 0:
            if console:
                console.print(f"[green]✅ Modèle {model} prêt[/green]")
            else:
                print_ok(f"Modèle {model} prêt")
            return True
        else:
            if console:
                console.print(f"[red]❌ Erreur pull modèle: {result.stderr}[/red]")
            else:
                print_err(f"Erreur pull modèle: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        if console:
            console.print("[red]❌ Timeout lors du téléchargement du modèle[/red]")
        else:
            print_err("Timeout lors du téléchargement du modèle")
        return False
    except Exception as e:
        if console:
            console.print(f"[red]❌ Erreur: {e}[/red]")
        else:
            print_err(f"Erreur: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    dirs = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "media_cache",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    if console:
        console.print("[green]✅ Dossiers créés[/green]")
    else:
        print_ok("Dossiers créés")
    return True


def setup_env_file():
    """Copy .env.example to .env if not exists"""
    if ENV_FILE.exists():
        if console:
            console.print("[yellow]⚠️  .env déjà existant[/yellow]")
        else:
            print_warn(".env déjà existant")
        return True

    if ENV_EXAMPLE.exists():
        shutil.copy(ENV_EXAMPLE, ENV_FILE)
        if console:
            console.print("[green]✅ .env créé depuis .env.example[/green]")
            console.print("[yellow]⚠️  IMPORTANT: Éditez .env avec vos vraies clés Twitter![/yellow]")
        else:
            print_ok(".env créé depuis .env.example")
            print_warn("IMPORTANT: Éditez .env avec vos vraies clés Twitter!")
        return True
    else:
        if console:
            console.print("[red]❌ .env.example introuvable[/red]")
        else:
            print_err(".env.example introuvable")
        return False


def verify_setup():
    """Verify the setup works"""
    if console:
        console.print("\n[cyan]🔍 Vérification de l'installation...[/cyan]")
    else:
        print_step("Vérification de l'installation")

    python_path = get_python_path()

    test_script = """
import tweepy
import yaml
import ollama
import sentence_transformers
import torch
import numpy
import requests
import langdetect
import apscheduler
import rich
import loguru
print("✅ Tous les imports OK")
"""
    try:
        result = subprocess.run([str(python_path), "-c", test_script], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            if console:
                console.print("[green]✅ Tous les modules importent correctement[/green]")
            else:
                print_ok("Tous les modules importent correctement")
            return True
        else:
            if console:
                console.print(f"[red]❌ Erreur imports: {result.stderr}[/red]")
            else:
                print_err(f"Erreur imports: {result.stderr}")
            return False
    except Exception as e:
        if console:
            console.print(f"[red]❌ Erreur vérification: {e}[/red]")
        else:
            print_err(f"Erreur vérification: {e}")
        return False


def print_banner():
    """Print setup banner"""
    banner = """
+==============================================================+
|  Barcelona Twitter Bot - Setup                               |
|  Configuration complete pour @UniversCule                    |
+==============================================================+
"""
    print(banner)


def main():
    print_banner()

    # Phase 1: Basic setup (no rich needed)
    steps = [
        ("Vérification Python", check_python_version),
        ("Création venv", create_venv),
        ("Installation dépendances", install_dependencies),
    ]

    for name, func in steps:
        print_step(name)
        if not func():
            print_err(f"Échec à l'étape: {name}")
            sys.exit(1)

    # Phase 2: Initialize rich now that deps are installed
    print_step("Initialisation interface riche")
    if not init_rich():
        print_err("Impossible d'initialiser rich")
        sys.exit(1)

    # Phase 3: Rich-enabled steps
    steps_rich = [
        ("Création dossiers", create_directories),
        ("Configuration .env", setup_env_file),
    ]

    for name, func in steps_rich:
        console.print(f"\n[bold]▶ {name}...[/bold]")
        if not func():
            console.print(f"\n[red]❌ Échec à l'étape: {name}[/red]")
            sys.exit(1)

    # Ollama setup
    console.print("\n[bold]▶ Configuration Ollama...[/bold]")
    if not check_ollama():
        console.print("[yellow]⚠️  Ollama non détecté[/yellow]")
        if platform.system() == "Windows":
            console.print("[cyan]📥 Installation automatique d'Ollama...[/cyan]")
            if install_ollama_windows():
                time.sleep(3)
            else:
                console.print("[yellow]⚠️  Installation Ollama échouée, installez manuellement[/yellow]")
        else:
            console.print("[yellow]💡 Installez Ollama: https://ollama.com/download[/yellow]")

    # Pull model if Ollama available
    if check_ollama():
        console.print("[cyan]📥 Téléchargement automatique du modèle qwen2.5:7b...[/cyan]")
        pull_ollama_model("qwen2.5:7b")

    # Final verification
    if verify_setup():
        console.print(Panel.fit(
            "[bold green]🎉 Setup terminé avec succès ![/bold green]\n\n"
            "Prochaines étapes :\n"
            "1. Vérifiez [cyan].env[/cyan] (vos clés Twitter sont déjà dedans)\n"
            "2. Lancez : [cyan].venv\\Scripts\\activate[/cyan] (Windows)\n"
            "3. Testez : [cyan]python -m src.main test[/cyan]\n"
            "4. Démarrez : [cyan]python -m src.main start[/cyan]",
            border_style="green"
        ))
    else:
        console.print("[yellow]⚠️  Setup terminé mais vérification partielle. Vérifiez les erreurs ci-dessus.[/yellow]")


if __name__ == "__main__":
    main()