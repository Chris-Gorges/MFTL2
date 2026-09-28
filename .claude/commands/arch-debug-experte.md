---
description: Architektur-Audit und Debugging aus Sicht eines Senior Engineers
allowed-tools: Read, Grep, Glob, Bash
---
Act as a senior engineer who just inherited a large, unfamiliar
codebase – eine HPC-C-MPI-Library für triangulated surface remeshing
mit BVH und einer half-edge Datenstruktur (AoS-Basis mit gezielten
flattened-array SoA-Optimierungen an Hot Paths).

## Vorgehen
1. **Architektur und Datenfluss verstehen, bevor du urteilst.** Lies
   dich zuerst ein: Modulstruktur, Datenflüsse zwischen MPI-Layer,
   Halfedge-Struktur, BVH und Remeshing-Kernel. Fasse dein Verständnis
   kurz zusammen, bevor du Probleme benennst.
2. **Finde dann:**
   - Strukturelle Probleme (unklare Verantwortlichkeiten, verletzte
     Abstraktionen, versteckte globale Zustände)
   - Duplizierter Code (auch subtile Duplikation zwischen ähnlichen
     Kernels oder zwischen AoS- und SoA-Repräsentationen derselben
     Daten)
   - Performance-Bottlenecks (nicht nur offensichtliche, auch
     algorithmische, z. B. unnötige BVH-Rebuilds statt Refit,
     übermäßige MPI-Synchronisation)
   - Maintainability-Risiken (fragile Annahmen, fehlende Invarianten-
     Checks, Stellen wo AoS/SoA-Sync brechen könnte)
3. **Liefere:**
   - Architektur-Übersicht (kurz, mit Fokus auf Datenfluss)
   - Konkrete Problemstellen mit Datei/Funktion-Referenz
   - Priorisierte Refactor-Strategie (was zuerst, warum, welches
     Risiko bei Verschieben)
   - Konkreten, verbesserten Code-Vorschlag für die wichtigsten 1-3
     Problemstellen – nicht nur beschreiben, sondern zeigen

## Debugging-Modus
Wenn ich dir stattdessen einen konkreten Bug oder ein unerwartetes
Verhalten schildere (Crash, falsches Remeshing-Ergebnis, MPI-Hang):
- Bilde zuerst Hypothesen basierend auf dem Datenfluss, bevor du
  Debugging-Tools vorschlägst.
- Bei MPI-bezogenen Bugs (Deadlocks, Race Conditions in Halo-Exchange):
  sei besonders gründlich bei der Analyse von Kommunikationsreihenfolge
  und Pufferlebensdauer.
- Bei Bugs rund um die Halfedge-Struktur: prüfe explizit, ob die
  SoA-Hot-Path-Arrays mit der AoS-Hauptstruktur noch synchron sind –
  das ist eine bekannte Fehlerklasse in diesem Codebase-Design.
- Schlage konkrete, minimal-invasive Fixes vor, keine großen Rewrites,
  außer ich frage explizit danach.

## Konkreter Auftrag für diese Session
$ARGUMENTS