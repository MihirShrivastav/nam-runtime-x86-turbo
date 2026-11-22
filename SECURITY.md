# Security policy

## Scope

namturbo reads model files from untrusted sources, so the loader is the main attack surface. Reportable issues:

- a model file (or any input to `Model::load`) that makes the library crash, hang, use a very large amount of memory, or read or write memory
  that it must not;
- any way in which `Processor::process` can be made to read or write outside its buffers.

The loader limits the file size (64 MB), checks every setting and every weight, and is tested with generated damaged files, but a test
cannot show that there is no defect.

## Reporting

Use the private vulnerability reporting of GitHub for this repository (**Security** tab, **Report a vulnerability**). Do
not put details in a public issue. If private reporting is not available, open an issue that asks for a private way to contact the
maintainer, without any details of the problem.

A report is most useful with: the version of the library, the operating system, the compiler, and the file or the sequence of calls that
causes the problem (a model file can be attached to the private report).

## Supported versions

The latest released version.
