import Link from "next/link";
import {
  ArrowRight,
  BrainCircuit,
  ChartNoAxesCombined,
  ChevronRight,
  Database,
  Dna,
  FileText,
  FlaskConical,
  Network,
  ShieldCheck,
} from "lucide-react";
import styles from "./page.module.css";

export default function Home() {
  return (
    <main className={styles.page}>
      <nav className={styles.nav} aria-label="Main navigation">
        <Link href="/" className={styles.brand}>
          <span className={styles.brandMark}><Dna size={20} strokeWidth={2.5} /></span>
          <span>Gene<span>Web</span></span>
        </Link>
        <div className={styles.navLinks}>
          <a href="#workflow">Workflow</a>
          <a href="#capabilities">Capabilities</a>
          <Link href="/login">Sign in <ArrowRight size={15} /></Link>
        </div>
      </nav>

      <section className={styles.hero}>
        <div className={styles.heroCopy}>
          <div className={styles.eyebrow}><span /> COMPUTATIONAL BIOLOGY WORKSPACE</div>
          <h1>From gene expression to <em>actionable</em> discovery.</h1>
          <p className={styles.heroLead}>
            A unified research environment for target identification, drug-target
            analysis, and AI-assisted affinity prediction.
          </p>
          <div className={styles.heroActions}>
            <Link href="/login?mode=signup" className={styles.primaryButton}>
              Create your workspace <ArrowRight size={17} />
            </Link>
            <Link href="/login" className={styles.textButton}>
              Sign in <ChevronRight size={17} />
            </Link>
          </div>
          <div className={styles.trustLine}><ShieldCheck size={16} /> Private, authenticated research workflows</div>
        </div>
        <div className={styles.heroVisual} aria-label="Platform workflow preview">
          <div className={styles.visualGlow} />
          <div className={styles.dataOrb}><Dna size={82} strokeWidth={1} /></div>
          <div className={`${styles.signalCard} ${styles.signalTop}`}><span className={styles.signalDot} /> TARGET IDENTIFICATION <strong>ACTIVE</strong></div>
          <div className={`${styles.signalCard} ${styles.signalBottom}`}><ChartNoAxesCombined size={18} /><span>Expression signal</span><strong>0.85</strong></div>
          <div className={styles.visualGrid} />
        </div>
      </section>

      <section id="workflow" className={styles.workflowSection}>
        <div className={styles.sectionHeading}>
          <span className={styles.sectionKicker}>ONE CONNECTED PIPELINE</span>
          <h2>Research moves in one direction.</h2>
          <p>Turn complex biological data into a clear path for investigation, without stitching together disconnected tools.</p>
        </div>
        <div className={styles.workflowGrid}>
          <WorkflowCard number="01" icon={<Database />} title="Identify targets" text="Upload gene-expression data, configure preprocessing, and train models through a guided interface." />
          <WorkflowCard number="02" icon={<Network />} title="Explore interactions" text="Connect expressed genes with drug associations and molecular representations from integrated datasets." />
          <WorkflowCard number="03" icon={<FlaskConical />} title="Prioritize candidates" text="Generate embeddings and predict drug-gene affinity to focus the next stage of discovery." />
        </div>
      </section>

      <section id="capabilities" className={styles.capabilityBand}>
        <div><BrainCircuit size={22} /><span>Configurable ML experiments</span></div>
        <div><Network size={22} /><span>DepMap-informed associations</span></div>
        <div><Dna size={22} /><span>Multi-modal embeddings</span></div>
        <div><ChartNoAxesCombined size={22} /><span>Reproducible artifacts</span></div>
      </section>

      <section className={styles.aboutSection}>
        <div className={styles.aboutCopy}>
          <span className={styles.sectionKicker}>HOW THE PLATFORM WORKS</span>
          <h2>A guided path from biological signal to candidate drug.</h2>
          <p>
            GeneWeb brings the computational drug-repurposing workflow into one
            workspace. Researchers upload gene-expression data, choose quality
            control and preprocessing steps, train and compare models, then use
            the resulting targets to explore drug associations and molecular
            representations.
          </p>
          <p>
            Long-running training, DepMap analysis, and embedding jobs run in
            background workers. Results, metrics, models, and downloadable
            artifacts are tracked so each stage can be inspected before moving
            to affinity prioritization.
          </p>
          <a href="/demonstration-mie.pdf" target="_blank" rel="noreferrer" className={styles.paperLink}>
            <FileText size={18} /> Read the platform demonstration paper <ArrowRight size={16} />
          </a>
        </div>
        <div className={styles.aboutSteps}>
          <div><span>01</span><strong>Target identification</strong><p>Configure preprocessing, select a model, validate performance, and rank relevant genes.</p></div>
          <div><span>02</span><strong>Drug-target exploration</strong><p>Connect targets to DepMap associations and retrieve compound information and representations.</p></div>
          <div><span>03</span><strong>Candidate prioritization</strong><p>Generate multi-modal embeddings and run drug-gene affinity inference for downstream review.</p></div>
        </div>
      </section>

      <footer className={styles.footer}>
        <span>GeneWeb</span><span>Built for computational biology research</span>
      </footer>
    </main>
  );
}

function WorkflowCard({ number, icon, title, text }: { number: string; icon: React.ReactNode; title: string; text: string }) {
  return (
    <article className={styles.workflowCard}>
      <div className={styles.cardTop}><span>{number}</span><div className={styles.cardIcon}>{icon}</div></div>
      <h3>{title}</h3>
      <p>{text}</p>
      <ChevronRight className={styles.cardArrow} size={18} />
    </article>
  );
}
