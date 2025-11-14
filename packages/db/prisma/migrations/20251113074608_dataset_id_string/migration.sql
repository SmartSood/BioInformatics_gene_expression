/*
  Warnings:

  - The primary key for the `Dataset` table will be changed. If it partially fails, the table could be left without primary key constraint.
  - You are about to drop the column `description` on the `Dataset` table. All the data in the column will be lost.
  - You are about to drop the column `fileSize` on the `Dataset` table. All the data in the column will be lost.
  - You are about to drop the column `fileType` on the `Dataset` table. All the data in the column will be lost.
  - You are about to drop the column `name` on the `Dataset` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "Dataset" DROP CONSTRAINT "Dataset_pkey",
DROP COLUMN "description",
DROP COLUMN "fileSize",
DROP COLUMN "fileType",
DROP COLUMN "name",
ALTER COLUMN "id" DROP DEFAULT,
ALTER COLUMN "id" SET DATA TYPE TEXT,
ADD CONSTRAINT "Dataset_pkey" PRIMARY KEY ("id");
DROP SEQUENCE "Dataset_id_seq";
