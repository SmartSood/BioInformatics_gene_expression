/*
  Warnings:

  - You are about to drop the column `metadata` on the `Dataset` table. All the data in the column will be lost.
  - Added the required column `description` to the `Dataset` table without a default value. This is not possible if the table is not empty.
  - Added the required column `name` to the `Dataset` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "Dataset" DROP COLUMN "metadata",
ADD COLUMN     "description" TEXT NOT NULL,
ADD COLUMN     "name" TEXT NOT NULL;
