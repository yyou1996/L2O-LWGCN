#!/bin/bash
# submit jobs

sbatch ./jobs/LWGCN_cora_job
sbatch ./jobs/LWGCN_pubmed_job
sbatch ./jobs/LWGCN_reddit_job
sbatch ./jobs/LWGCN_amazon_670k_job
sbatch ./jobs/LWGCN_amazon_3m_job

sbatch ./jobs/L2O_LWGCN_cora_job
sbatch ./jobs/L2O_LWGCN_pubmed_job
sbatch ./jobs/L2O_LWGCN_reddit_job
sbatch ./jobs/L2O_LWGCN_amazon_670k_job
sbatch ./jobs/L2O_LWGCN_amazon_3m_job
