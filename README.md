# Multimodal Geospatial RAG Agent for Environmental Impact Assessments (EIA)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: Geoinformatics & Agentic AI](https://img.shields.io/badge/Domain-Geoinformatics%20%7C%20Agentic%20AI-green.svg)](#)
[![Stack: LangChain | ChromaDB | Groq](https://img.shields.io/badge/Stack-LangChain%20%7C%20ChromaDB%20%7C%20Groq-orange.svg)](#)
[![LLM: LLaMA--3.1--70B](https://img.shields.io/badge/LLM-LLaMA--3.1--70B-blueviolet.svg)](#)

An autonomous Multimodal Retrieval-Augmented Generation (RAG) agent bridging Earth observation satellite telemetry with statutory environmental jurisprudence. The system indexes environmental protection acts, protected wildlife corridors, and riparian buffer statutes into a **ChromaDB** vector store using **HuggingFace** dense embeddings. It cross-references remote sensing raster metrics (canopy fragmentation, deforestation hectares) to synthesize legal compliance verdicts via **LLaMA-3.1-70B on Groq**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Problem Statement & Background](#problem-statement--background)
- [System Architecture](#system-architecture)
- [Mathematical & Semantic Methodology](#mathematical--semantic-methodology)
- [Statutory Knowledge Base & Spatial Telemetry](#statutory-knowledge-base--spatial-telemetry)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Execution & Usage](#execution--usage)
- [Audit Output Sample](#audit-output-sample)
- [License](#license)

---

## Project Overview

Statutory Environmental Impact Assessments (EIA) require environmental regulators and natural resources engineers to reconcile quantitative geospatial statistics (e.g., tree canopy loss from satellite imagery, proximity to riverbanks) with dense, multi-statute environmental legal codes.

This platform automates this clearance workflow:
1. **Vectorized Regulatory Knowledge Base:** Indexes complex environmental policies (Forest Conservation Acts, Riparian Ecology Directives, CAMPA Afforestation Ratios) into **ChromaDB** using **HuggingFace `all-MiniLM-L6-v2`** embeddings.
2. **Spatial Telemetry Ingestion:** Extracts multi-layer spatial attributes from satellite rasters (Hansen 30m Global Forest Change) and vector geometries.
3. **Agentic Legal Reasoning:** Prompts **LLaMA-3.1-70B via Groq API** with semantic context and spatial telemetry to generate legal audit reports, cite specific statutory violations, and compute mandatory mitigation quotas.

---

## Key Features

- **Dense Semantic Retrieval:** Employs cosine-similarity vector search over chunked environmental jurisprudence using LangChain and ChromaDB.
- **Ultra-Low Latency Inference:** Integrates **ChatGroq** running LLaMA-3.1-70B for near-instantaneous synthesis of multi-source audit reports.
- **Multimodal Context Merging:** Ingests quantitative spatial telemetry (canopy %, distance in meters, deforested hectares) directly into semantic LLM reasoning chains.
- **Compensatory Calculator:** Computes exact compensatory afforestation ratios (1:2 non-forest / 1:1 degraded forest) directly from spatial clearance metrics.
- **Fail-Safe Execution:** Includes a local deterministic fallback mode to allow local testing even without an active Groq API key.

---

## Problem Statement & Background

Environmental governance faces two major bottlenecks:

1. **Information Silos:** Spatial analytics teams produce raster maps (GIS/Remote Sensing), while legal compliance officers interpret policy PDFs. The lack of automated synthesis leads to clearance delays or missed ecological violations.
2. **Hallucination Risks in Generic LLMs:** Standard large language models lack grounded context for local environmental laws. Implementing a **Domain-Specific RAG Architecture** ensures all regulatory decisions are backed by statutory citations.

---

## System Architecture

```text
  ┌─────────────────────────────────┐        ┌──────────────────────────────────┐
  │  Environmental Statutory Acts   │        │  Hansen Forest Change & Vectors  │
  │  (FCA, CAMPA, Riparian Buffers) │        │   (Spatial Zonal Raster Stats)   │
  └────────────────┬────────────────┘        └─────────────────┬────────────────┘
                   │                                           │
                   ▼                                           │
  ┌─────────────────────────────────┐                          │
  │  Recursive Character Splitting  │                          │
  └────────────────┬────────────────┘                          │
                   │                                           │
                   ▼                                           │
  ┌─────────────────────────────────┐                          │
  │ HuggingFace MiniLM Embeddings   │                          │
  └────────────────┬────────────────┘                          │
                   │                                           │
                   ▼                                           │
  ┌─────────────────────────────────┐                          │
  │  ChromaDB Vector Store Index    │                          │
  └────────────────┬────────────────┘                          │
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │    Context & Spatial Telemetry Merging    │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │     Groq Cloud Inference Engine           │
                   │         (LLaMA-3.1-70B Versatile)         │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   Formal Environmental Clearance Report   │
                   │  [Verdict | Violations | Mitigation Mand] │
                   └───────────────────────────────────────────┘
