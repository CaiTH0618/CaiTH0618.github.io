---
layout: page
title: PowerShell Tricks
---

### Create a symbolic link in Windows

Open PowerShell in administrator mode and run
```powershell
New-Item -ItemType SymbolicLink -Path "C:\path\to\link\file.txt" -Target "C:\path\to\target\file.txt"
```
