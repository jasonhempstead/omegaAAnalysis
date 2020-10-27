#include <cstddef>
#include <iostream>

#include "TROOT.h"
#include "TH3.h"

constexpr int errHistNotFound = -1;
constexpr int errArrayWrongSize = -2;

extern "C" {
int hist3dToNumpyArray(double* data, std::size_t size, const char* histName);
int numpyArrayToHist3d(const double* data, std::size_t size,
                       const char* histName);
int fillHist3dErrors(const double* data, std::size_t size,
                     const char* histName);
}

int checkHist(std::size_t size, const TH3* hist) {
  if (hist == nullptr) {
    return errHistNotFound;
  }

  std::size_t nHistBins =
      hist->GetNbinsX() * hist->GetNbinsY() * hist->GetNbinsZ();
  if (nHistBins != size) {
    return errArrayWrongSize;
  }

  return 0;
}

int hist3dToNumpyArray(double* data, std::size_t size, const char* histName) {
  const TH3* hist = (TH3*)gROOT->FindObject(histName);
  int retcode = checkHist(size, hist);
  if (retcode != 0) {
    return retcode;
  }

  for (int zBin = 1; zBin <= hist->GetNbinsZ(); ++zBin) {
    for (int yBin = 1; yBin <= hist->GetNbinsY(); ++yBin) {
      for (int xBin = 1; xBin <= hist->GetNbinsX(); ++xBin) {
        *data++ = hist->GetBinContent(xBin, yBin, zBin);
      }
    }
  }

  return 0;
}

int numpyArrayToHist3d(const double* data, std::size_t size,
                       const char* histName) {
  TH3* hist = (TH3*)gROOT->FindObject(histName);
  int retcode = checkHist(size, hist);
  if (retcode != 0) {
    return retcode;
  }

  for (int zBin = 1; zBin <= hist->GetNbinsZ(); ++zBin) {
    for (int yBin = 1; yBin <= hist->GetNbinsY(); ++yBin) {
      for (int xBin = 1; xBin <= hist->GetNbinsX(); ++xBin) {
        hist->SetBinContent(hist->GetBin(xBin, yBin, zBin), *data++);
      }
    }
  }

  return 0;
}

int fillHist3dErrors(const double* data, std::size_t size,
                     const char* histName) {
  TH3* hist = (TH3*)gROOT->FindObject(histName);
  int retcode = checkHist(size, hist);
  if (retcode != 0) {
    return retcode;
  }

  for (int zBin = 1; zBin <= hist->GetNbinsZ(); ++zBin) {
    for (int yBin = 1; yBin <= hist->GetNbinsY(); ++yBin) {
      for (int xBin = 1; xBin <= hist->GetNbinsX(); ++xBin) {
        hist->SetBinError(hist->GetBin(xBin, yBin, zBin), *data++);
      }
    }
  }

  return 0;
}