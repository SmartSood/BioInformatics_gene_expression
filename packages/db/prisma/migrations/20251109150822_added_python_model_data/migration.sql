-- CreateTable
CREATE TABLE "TrainingRun" (
    "id" TEXT NOT NULL,
    "userId" INTEGER NOT NULL,
    "status" TEXT NOT NULL,
    "datasetUri" TEXT NOT NULL,
    "modelPath" TEXT,
    "metrics" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "TrainingRun_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "TrainingRun" ADD CONSTRAINT "TrainingRun_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
